"""
Phase 4 - end-to-end resolution: block -> score -> match -> cluster -> evaluate.

Reports END-TO-END numbers. Phase 3's F1 of 0.88 was conditional on blocking
having surfaced the pair; multiply it by blocking's 68% recall and the real
figure was ~0.65 recall. Everything below is measured against the full set of
true cross-platform pairs, with no such conditioning.

Two metric families, because they fail differently:

  pairwise P/R/F1  - the metric resolution.py reports today. It tolerates
                     over-merging badly: one giant wrong cluster still scores
                     many pairs correctly.
  B-cubed P/R/F1   - the standard entity-resolution cluster metric. Scores
                     each ACCOUNT by how pure and how complete its assigned
                     cluster is, so a single over-merged blob is penalised
                     once per account inside it. This is the honest one.

Three configurations are compared so each fix is attributable:
  A) threshold + connected components   (what resolution.py does today)
  B) + constrained 1-1 matching
  C) + cluster hygiene
"""
import os
import sys
import time
from collections import defaultdict

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src import blocking, clustering  # noqa: E402
from src.config import settings  # noqa: E402
from src.models import text_model  # noqa: E402
from src.models.fusion_model import FellegiSunterFusion  # noqa: E402
from src.models.pair_features import AccountUniverse, extract_pair_features  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
COLLECTION_NAME = "identity_accounts"
MODEL_PATH = os.path.join(DATA_DIR, "fusion_model.json")


def bcubed(clusters, truth_by_record: dict, score_records: set | None = None) -> dict:
    """B-cubed precision/recall/F1 over accounts.

    `score_records` restricts which accounts are AVERAGED over, without
    changing the clusters or the truth groups they are scored against. That
    distinction matters for held-out evaluation: shrinking the clusters to a
    record subset first would delete every impurity that came from outside the
    subset and report a purity the system never achieved.
    """
    cluster_of = {}
    for idx, cluster in enumerate(clusters):
        for rid in cluster:
            cluster_of[rid] = idx
    cluster_members = {idx: c for idx, c in enumerate(clusters)}

    truth_members = defaultdict(set)
    for rid, eid in truth_by_record.items():
        truth_members[eid].add(rid)

    scored = truth_by_record if score_records is None else {
        rid: eid for rid, eid in truth_by_record.items() if rid in score_records}

    precisions, recalls = [], []
    for rid, eid in scored.items():
        predicted = cluster_members[cluster_of[rid]]
        actual = truth_members[eid]
        overlap = len(predicted & actual)
        precisions.append(overlap / len(predicted))
        recalls.append(overlap / len(actual))

    p = float(np.mean(precisions))
    r = float(np.mean(recalls))
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def pairwise_metrics(clusters, truth_by_record: dict, n_true_pairs: int) -> dict:
    """Pairwise TP/FP/FN computed from cluster composition instead of by
    materialising the pair sets.

    Enumerating every within-cluster pair is O(size^2), and over-merging is
    exactly the failure mode being measured here - a single runaway cluster of
    a few thousand accounts would mean millions of tuples. Counting
    combinatorially is exact and linear:
        TP = sum over clusters, over entities inside them, of C(k, 2)
    Every account of a person sits on a different platform, so all within-
    entity pairs are cross-platform and comparable to the ground-truth set.
    """
    tp = 0
    predicted_total = 0
    for cluster in clusters:
        size = len(cluster)
        predicted_total += size * (size - 1) // 2
        by_entity = defaultdict(int)
        for rid in cluster:
            by_entity[truth_by_record[rid]] += 1
        for k in by_entity.values():
            tp += k * (k - 1) // 2

    fp = predicted_total - tp
    fn = n_true_pairs - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def cluster_stats(clusters) -> dict:
    sizes = [len(c) for c in clusters]
    return {
        "n_clusters": len(clusters),
        "singletons": sum(1 for s in sizes if s == 1),
        "largest": max(sizes) if sizes else 0,
        "mean_size": float(np.mean(sizes)) if sizes else 0.0,
    }


def report(name, clusters, n_true_pairs, truth_by_record):
    pw = pairwise_metrics(clusters, truth_by_record, n_true_pairs)
    bc = bcubed(clusters, truth_by_record)
    st = cluster_stats(clusters)
    print(f"\n  {name}")
    print(f"    pairwise : P={pw['precision']:.4f} R={pw['recall']:.4f} F1={pw['f1']:.4f}  "
          f"(tp={pw['tp']} fp={pw['fp']} fn={pw['fn']})")
    print(f"    B-cubed  : P={bc['precision']:.4f} R={bc['recall']:.4f} F1={bc['f1']:.4f}")
    print(f"    clusters : {st['n_clusters']} ({st['singletons']} singletons), "
          f"largest={st['largest']}, mean size={st['mean_size']:.2f}")
    return {"pairwise": pw, "bcubed": bc, "stats": st}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()

    print("Loading vectors + account universe...")
    client = QdrantClient(settings.qdrant_url, timeout=120.0)
    vectors = text_model.load_vectors_from_qdrant(client, COLLECTION_NAME)
    universe = AccountUniverse.load(DATA_DIR, vectors=vectors)
    print(f"  {len(universe)} accounts, {len(vectors)} vectors")

    gt = pd.read_parquet(os.path.join(DATA_DIR, "ground_truth_pairs.parquet"))
    true_pairs = {tuple(sorted((r.record_id_a, r.record_id_b))) for r in gt.itertuples(index=False)}
    truth_by_record = {rid: acct["entity_id"] for rid, acct in universe.accounts.items()}
    print(f"  {len(true_pairs)} true cross-platform pairs")

    print("\n=== Blocking (multi-path) ===")
    t = time.time()
    candidates = blocking.generate_candidates(universe, vectors=vectors)
    print(f"  {len(candidates)} candidate pairs in {time.time() - t:.1f}s")
    found = int(candidates["label"].sum())
    print(f"  BLOCKING RECALL: {found}/{len(true_pairs)} = {found / len(true_pairs):.1%}")
    print(f"  reduction ratio: {len(candidates) / (len(universe) ** 2 / 2):.2e} of all possible pairs")

    print("\n  Per-path contribution:")
    contrib = blocking.path_contributions(candidates, len(true_pairs))
    print(f"    {'path':<14} {'candidates':>11} {'recall':>8} {'unique':>8}")
    for row in contrib.itertuples(index=False):
        print(f"    {row.path:<14} {row.candidates:>11} {row.recall:>7.1%} {row.unique_true_pairs:>8}")

    print("\n=== Scoring with the Phase 3 fusion model ===")
    model = FellegiSunterFusion.load(MODEL_PATH)
    t = time.time()
    pair_list = [(r.record_id_a, r.record_id_b) for r in candidates.itertuples(index=False)]
    feature_rows = [extract_pair_features(a, b, universe) for a, b in pair_list]
    scores = model.predict_proba(feature_rows)
    print(f"  scored {len(scores)} pairs in {time.time() - t:.1f}s")

    threshold = settings.fusion_match_threshold
    print(f"  threshold = {threshold} (settings.fusion_match_threshold)")

    all_ids = list(universe.accounts.keys())

    print("\n=== Configuration comparison ===")
    results = {}

    edges_a = clustering.threshold_edges(pair_list, scores, threshold)
    clusters_a, _ = clustering.build_clusters(edges_a, all_ids)
    results["A"] = report("A) threshold + connected components  [today's resolution.py]",
                          clusters_a, len(true_pairs), truth_by_record)

    edges_b = clustering.constrained_match(pair_list, scores, universe, threshold)
    clusters_b, graph_b = clustering.build_clusters(edges_b, all_ids)
    results["B"] = report("B) + constrained 1-1 matching", clusters_b, len(true_pairs), truth_by_record)

    clusters_c = clustering.enforce_cluster_hygiene(clusters_b, graph_b, universe)
    results["C"] = report("C) + cluster hygiene", clusters_c, len(true_pairs), truth_by_record)

    # The threshold that maximises PAIRWISE F1 is not the one that maximises
    # end-to-end clustering quality. Pairwise scoring judges each pair in
    # isolation, but in clustering a single false edge welds two identities
    # together and costs far more than one wrong pair - so the clustering
    # stage wants a stricter threshold than the classifier's own optimum.
    # Sweeping on the objective we actually care about:
    print("\n=== Threshold sweep on the end-to-end objective (config C) ===")
    print(f"  {'thr':>5} {'pair P':>8} {'pair R':>8} {'pair F1':>8} {'B3 P':>8} {'B3 R':>8} {'B3 F1':>8} {'largest':>8}")
    sweep = []
    for thr in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95]:
        edges = clustering.constrained_match(pair_list, scores, universe, thr)
        clusters_t, graph_t = clustering.build_clusters(edges, all_ids)
        clusters_t = clustering.enforce_cluster_hygiene(clusters_t, graph_t, universe)
        pw = pairwise_metrics(clusters_t, truth_by_record, len(true_pairs))
        bc = bcubed(clusters_t, truth_by_record)
        st = cluster_stats(clusters_t)
        sweep.append((thr, pw, bc, st))
        print(f"  {thr:>5.2f} {pw['precision']:>8.4f} {pw['recall']:>8.4f} {pw['f1']:>8.4f} "
              f"{bc['precision']:>8.4f} {bc['recall']:>8.4f} {bc['f1']:>8.4f} {st['largest']:>8}")
    best_thr, best_pw, best_bc, _ = max(sweep, key=lambda s: s[2]["f1"])
    print(f"\n  Best end-to-end B-cubed F1: {best_bc['f1']:.4f} at threshold {best_thr:.2f} "
          f"(pairwise F1 {best_pw['f1']:.4f})")
    print("  NOTE: this sweep runs over all entities, so it is an operating-point")
    print("  curve, not a held-out result. A stricter protocol would pick the")
    print("  threshold on held-out entities only.")

    print("\n=== Summary ===")
    print(f"  {'config':<34} {'pair F1':>9} {'B3 F1':>9} {'largest':>9}")
    labels = {"A": "A) threshold + conn. components", "B": "B) + constrained matching", "C": "C) + hygiene"}
    for key in ("A", "B", "C"):
        r = results[key]
        print(f"  {labels[key]:<34} {r['pairwise']['f1']:>9.4f} {r['bcubed']['f1']:>9.4f} "
              f"{r['stats']['largest']:>9}")

    print(f"\nTotal runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
