"""
Phase 5 - train and evaluate the virtual -> real identity mapper.

Two evaluations, deliberately separated:

  A) GIVEN PERFECT CLUSTERS - clusters taken straight from ground-truth
     entity groupings. Isolates the mapper's own quality from Phase 4's
     clustering errors.

  B) END TO END - clusters as Phase 4 actually predicts them. The difference
     between A and B is the cost that clustering errors impose on the final
     answer, which is the number a client should be quoted, not A.

Within each, accuracy is broken out for entities that have a TWIN (a distinct
person sharing name + city + birth year) versus those that don't. Twins can
only be separated by job/education/phone/email, which exist almost solely on
LinkedIn and Telegram - so this split shows precisely where the method stops
working, rather than hiding it inside an average.

Splits are by entity_id into three disjoint groups (fit / calibration / test),
same discipline as Phase 3: no person contributes to more than one split.
"""
import json
import os
import sys
import time
from collections import defaultdict

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

from src import blocking, clustering
from src.config import settings
from src.models import text_model
from src.models.fusion_model import FellegiSunterFusion, calibration_report
from src.models.pair_features import AccountUniverse, extract_pair_features
from src.real_identity_mapper import (RealIdentityMapper, RegistryIndex,
                                      candidates_for_clusters, extract_cluster_features,
                                      fit_top1_calibration)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
COLLECTION_NAME = "identity_accounts"
PAIR_MODEL_PATH = os.path.join(DATA_DIR, "fusion_model.json")
MAPPER_MODEL_PATH = os.path.join(DATA_DIR, "mapper_model.json")
CLUSTERS_PATH = os.path.join(DATA_DIR, "resolution_clusters.parquet")
IDENTITY_MAP_PATH = os.path.join(DATA_DIR, "identity_map.parquet")
TOP1_CALIBRATION_PATH = os.path.join(DATA_DIR, "mapper_top1_calibration.json")


def truth_clusters(universe) -> tuple[list[set], list[str]]:
    """Ground-truth account groupings, one per entity present in the data."""
    groups = defaultdict(set)
    for rid, acct in universe.accounts.items():
        groups[acct["entity_id"]].add(rid)
    entity_ids = sorted(groups)
    return [groups[e] for e in entity_ids], entity_ids


def predicted_clusters(universe, vectors):
    """Re-run the Phase 4 pipeline to get the clusters the system actually
    produces (blocking -> fusion scoring -> constrained match -> hygiene)."""
    candidates = blocking.generate_candidates(universe, vectors=vectors, verbose=False)
    pair_list = [(r.record_id_a, r.record_id_b) for r in candidates.itertuples(index=False)]
    model = FellegiSunterFusion.load(PAIR_MODEL_PATH)
    rows = [extract_pair_features(a, b, universe) for a, b in pair_list]
    scores = model.predict_proba(rows)

    # Phase 4's sweep put the best end-to-end B-cubed F1 (0.864) at the same
    # 0.45 as the pairwise optimum, so no separate clustering threshold.
    threshold = settings.fusion_match_threshold
    edges = clustering.constrained_match(pair_list, scores, universe, threshold)
    clusters, graph = clustering.build_clusters(edges, list(universe.accounts.keys()))
    clusters = clustering.enforce_cluster_hygiene(clusters, graph, universe)
    return clusters


def cluster_true_entity(cluster, universe) -> str:
    """Majority entity in a cluster - its 'real' owner for scoring purposes.

    A two-way tie is common (a cluster that merged exactly one account from
    each of two people), so the tie is broken on the entity id. Taking
    max() over an unordered set would resolve it by string hash, which differs
    per process and moved the reported cluster counts and accuracy between
    otherwise identical runs.
    """
    counts = defaultdict(int)
    for rid in sorted(cluster):
        counts[universe.accounts[rid]["entity_id"]] += 1
    return max(sorted(counts.items()), key=lambda kv: kv[1])[0]


def build_training_rows(clusters, universe, index, candidate_lists):
    rows, labels, meta = [], [], []
    for cluster, candidates in zip(clusters, candidate_lists):
        true_entity = cluster_true_entity(cluster, universe)
        members = [universe.accounts[rid] for rid in cluster]
        for entity_id in candidates:
            rows.append(extract_cluster_features(members, index.records[entity_id]))
            labels.append(int(entity_id == true_entity))
            meta.append({"true_entity": true_entity, "candidate": entity_id})
    return rows, labels, meta


def evaluate(clusters, universe, index, mapper, twin_entities, label: str,
             confidence_threshold: float = 0.5):
    """Top-1 / top-5 accuracy over clusters, split by twin status."""
    t = time.time()
    candidate_lists = candidates_for_clusters(clusters, universe, index)

    stats = {"all": [0, 0, 0, 0, 0], "twin": [0, 0, 0, 0, 0], "non_twin": [0, 0, 0, 0, 0]}
    #        [n, top1, top5, confident_and_correct, answered]
    unreachable = 0
    calib_probs, calib_labels = [], []

    for cluster, candidates in zip(clusters, candidate_lists):
        true_entity = cluster_true_entity(cluster, universe)
        members = [universe.accounts[rid] for rid in cluster]
        ranked = mapper.rank(members, candidates, top_n=5)

        if true_entity not in candidates:
            unreachable += 1

        if ranked:
            calib_probs.append(ranked[0]["confidence_top1"] or ranked[0]["confidence"])
            calib_labels.append(int(ranked[0]["entity_id"] == true_entity))

        group = "twin" if true_entity in twin_entities else "non_twin"
        top1_conf = (ranked[0]["confidence_top1"] or ranked[0]["confidence"]) if ranked else 0.0
        answered = bool(ranked) and top1_conf >= confidence_threshold
        for key in ("all", group):
            stats[key][0] += 1
            if ranked and ranked[0]["entity_id"] == true_entity:
                stats[key][1] += 1
            if any(r["entity_id"] == true_entity for r in ranked):
                stats[key][2] += 1
            if answered and ranked[0]["entity_id"] == true_entity:
                stats[key][3] += 1
            if answered:
                stats[key][4] += 1

    print(f"\n  {label}  ({len(clusters)} clusters, {time.time() - t:.1f}s)")
    print(f"    {'group':<10} {'n':>6} {'top-1':>8} {'top-5':>8} {'answered':>10} {'prec|ans':>9}")
    for key in ("all", "non_twin", "twin"):
        n, top1, top5, confident_correct, answered = stats[key]
        if not n:
            continue
        # Precision among the cases the system was willing to answer - the
        # number that matters operationally, since an abstention is cheap and
        # a confident wrong attribution to a real person is not.
        prec = confident_correct / answered if answered else float("nan")
        print(f"    {key:<10} {n:>6} {top1 / n:>7.1%} {top5 / n:>7.1%} "
              f"{answered / n:>9.1%} {prec:>8.1%}")
    print(f"    blocking missed the true entity for {unreachable} clusters "
          f"({unreachable / max(1, len(clusters)):.1%})")

    # Calibration on HELD-OUT clusters: does "87% confident" actually mean the
    # top-1 attribution is right 87% of the time? Measuring this on the
    # calibration split itself would be circular (isotonic regression fits it
    # exactly by construction), so it is only meaningful here on test data.
    if calib_probs:
        print(f"    top-1 confidence vs. observed correctness:")
        print(f"      {'bin':>10} {'predicted':>10} {'observed':>10} {'n':>7}")
        for bucket in calibration_report(np.array(calib_probs), np.array(calib_labels)):
            print(f"      {bucket['bin']:>10} {bucket['predicted']:>10.3f} "
                  f"{bucket['observed']:>10.3f} {bucket['n']:>7}")
    return stats


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()

    print("Loading universe + registry...")
    client = QdrantClient(settings.qdrant_url, timeout=120.0)
    vectors = text_model.load_vectors_from_qdrant(client, COLLECTION_NAME)
    universe = AccountUniverse.load(DATA_DIR, vectors=vectors)
    index = RegistryIndex.load(DATA_DIR)
    print(f"  {len(universe)} accounts, {len(index)} registry records")

    registry_df = pd.read_parquet(os.path.join(DATA_DIR, "real_identity_registry.parquet"))
    twin_entities = set(registry_df[registry_df["twin_of"].notna()]["entity_id"])
    twin_entities |= set(registry_df[registry_df["twin_of"].notna()]["twin_of"].dropna())
    print(f"  {len(twin_entities)} entities involved in a twin pair")

    gt_clusters, gt_entities = truth_clusters(universe)
    print(f"  {len(gt_clusters)} ground-truth clusters")

    # entity-disjoint three-way split
    rng = np.random.default_rng(42)
    order = rng.permutation(len(gt_entities))
    n_fit, n_cal = int(0.5 * len(order)), int(0.2 * len(order))
    split_of = {}
    for pos, idx in enumerate(order):
        split_of[gt_entities[idx]] = "fit" if pos < n_fit else ("cal" if pos < n_fit + n_cal else "test")

    print("\nBuilding candidates for ground-truth clusters...")
    t = time.time()
    candidate_lists = candidates_for_clusters(gt_clusters, universe, index)
    avg = sum(len(c) for c in candidate_lists) / len(candidate_lists)
    hit = sum(1 for cl, cands in zip(gt_clusters, candidate_lists)
              if cluster_true_entity(cl, universe) in cands)
    print(f"  {avg:.1f} candidates/cluster, blocking recall {hit / len(gt_clusters):.1%} "
          f"in {time.time() - t:.1f}s")

    print("\nTraining the mapper (Fellegi-Sunter, entity-disjoint splits)...")
    by_split = {"fit": [], "cal": [], "test": []}
    for cluster, candidates in zip(gt_clusters, candidate_lists):
        by_split[split_of[cluster_true_entity(cluster, universe)]].append((cluster, candidates))

    model = FellegiSunterFusion()
    for split in ("fit", "cal"):
        clusters_s = [c for c, _ in by_split[split]]
        cands_s = [c for _, c in by_split[split]]
        rows, labels, _ = build_training_rows(clusters_s, universe, index, cands_s)
        if split == "fit":
            model.fit(rows, labels)
            print(f"  fit on {len(rows)} cluster-candidate rows ({sum(labels)} positive)")
        else:
            model.fit_calibration(rows, labels)
            print(f"  calibrated on {len(rows)} rows")

    model.save(MAPPER_MODEL_PATH)
    print(f"  saved -> {MAPPER_MODEL_PATH}")

    print("\nRunning Phase 4 resolution to get the clusters the system really produces...")
    t = time.time()
    pred_clusters = predicted_clusters(universe, vectors)
    print(f"  {len(pred_clusters)} predicted clusters in {time.time() - t:.1f}s")
    pred_by_split = defaultdict(list)
    for cluster in pred_clusters:
        pred_by_split[split_of.get(cluster_true_entity(cluster, universe))].append(cluster)

    # Second calibration stage, asking the different question the dossier
    # actually displays: given the top-ranked candidate scored X, how often is
    # it the right person?
    #
    # Fitted on PREDICTED clusters, not ground-truth ones. The deployed system
    # only ever sees predicted clusters, and those carry upstream clustering
    # errors that a calibrator fitted on perfect clusters knows nothing about -
    # fitting on the clean case made the top bin claim 99.8% where the deployed
    # path actually delivered 90.7%.
    mapper = RealIdentityMapper(index, model)
    cal_clusters = pred_by_split["cal"]
    cal_candidates = candidates_for_clusters(cal_clusters, universe, index)
    cal_conf, cal_share, cal_correct = [], [], []
    for cluster, candidates in zip(cal_clusters, cal_candidates):
        members = [universe.accounts[rid] for rid in cluster]
        ranked = mapper.rank(members, candidates, top_n=1)
        if ranked:
            cal_conf.append(ranked[0]["confidence"])
            cal_share.append(ranked[0]["share"])
            cal_correct.append(int(ranked[0]["entity_id"] == cluster_true_entity(cluster, universe)))
    top1_cal = fit_top1_calibration(cal_conf, cal_share, cal_correct)
    with open(TOP1_CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(top1_cal, f)
    print(f"  top-1 calibration fitted on {len(cal_conf)} held-out predicted clusters "
          f"({sum(cal_correct) / max(1, len(cal_correct)):.1%} of them correct) -> {TOP1_CALIBRATION_PATH}")
    mapper = RealIdentityMapper(index, model, top1_calibration=top1_cal)

    print("\n=== A) Mapper quality given PERFECT clusters (test split only) ===")
    test_clusters = [c for c, _ in by_split["test"]]
    evaluate(test_clusters, universe, index, mapper, twin_entities, "ground-truth clusters")

    print("\n=== B) End to end: clusters as Phase 4 predicts them ===")
    evaluate(pred_by_split["test"], universe, index, mapper, twin_entities,
             "predicted clusters (test entities)")

    print("\nPersisting full resolution output for the API...")
    cluster_rows = []
    for cluster_id, cluster in enumerate(pred_clusters):
        for rid in sorted(cluster):
            cluster_rows.append({"cluster_id": cluster_id, "record_id": rid})
    pd.DataFrame(cluster_rows).to_parquet(CLUSTERS_PATH, index=False)
    print(f"  {len(cluster_rows)} account->cluster rows -> {CLUSTERS_PATH}")

    all_candidates = candidates_for_clusters(pred_clusters, universe, index)
    map_rows = []
    for cluster_id, (cluster, candidates) in enumerate(zip(pred_clusters, all_candidates)):
        members = [universe.accounts[rid] for rid in cluster]
        ranked = mapper.rank(members, candidates, top_n=3)
        for rank, entry in enumerate(ranked):
            map_rows.append({
                "cluster_id": cluster_id,
                "rank": rank,
                "entity_id": entry["entity_id"],
                "confidence": entry["confidence"],
                # what the dossier displays for the top-ranked attribution
                "confidence_top1": entry["confidence_top1"],
                "share": entry["share"],
                "full_name": entry["registry"]["full_name"],
                "city": entry["registry"]["city"],
                "birth_year": entry["registry"]["birth_year"],
                "job": entry["registry"]["job"],
                "education": entry["registry"]["education"],
            })
    pd.DataFrame(map_rows).to_parquet(IDENTITY_MAP_PATH, index=False)
    print(f"  {len(map_rows)} cluster->identity rows -> {IDENTITY_MAP_PATH}")

    print(f"\nTotal runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
