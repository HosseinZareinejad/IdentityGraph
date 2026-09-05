"""
Phase 6 - the consolidated evaluation. Everything a reader needs to judge the
system, computed in one run and written to data/metrics.json.

The other scripts each train something and report on it. This one trains
nothing: it loads the shipped models and measures them, so the numbers it
prints are the numbers the deployed system produces.

Four things here are NOT in the earlier phases, and they are the point of this
script:

  BLOCKING PATH ECONOMICS. Recall is meaningless without cost. Each path is
  removed in turn to show the recall it uniquely carries and the candidates it
  charges for that. A path that adds 0.1% recall for 27% of the candidate set
  should be known to be doing that.

  A HONEST THRESHOLD PROTOCOL. run_resolution.py sweeps the operating
  threshold over every entity, which selects a hyperparameter on the data used
  to report the result. Here the sweep runs on validation entities and the
  chosen value is applied, once, to test entities.

  DEGRADATION CURVES. A single accuracy number hides the only question an
  operator actually asks: how much do I need to know about someone before this
  works? Accuracy is reported against cluster size, platform mix, and which
  identifying fields the cluster happens to carry.

  ERROR TAXONOMY. Every end-to-end miss is attributed to the stage that caused
  it - blocking never surfaced the person, clustering split or merged them, or
  the ranker had them and chose wrong. Without this the three failure modes
  are indistinguishable in the headline number, and they need completely
  different fixes.
"""
import json
import os
import sys
import time
from collections import Counter

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src import blocking, clustering  # noqa: E402
from src.config import settings  # noqa: E402
from src.models import text_model  # noqa: E402
from src.models.fusion_model import FellegiSunterFusion, brier_score, calibration_report  # noqa: E402
from src.models.pair_features import AccountUniverse, extract_pair_features  # noqa: E402
from src.real_identity_mapper import (RealIdentityMapper, RegistryIndex,  # noqa: E402
                                      candidates_for_clusters)
from scripts.run_resolution import bcubed, cluster_stats, pairwise_metrics  # noqa: E402
from scripts.train_mapper import cluster_true_entity, truth_clusters  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
COLLECTION_NAME = "identity_accounts"
METRICS_PATH = os.path.join(DATA_DIR, "metrics.json")

OPTIONAL_PATHS = ["phonetic", "contact", "name_fuzzy", "username", "username_skel",
                  "cross_field", "vector_bio", "vector_posts"]
IDENTIFYING_FIELDS = ["city", "birth_year", "job_title", "education",
                      "phone_canonical", "email_normalized"]


# ---------------------------------------------------------------------------
# 1. Corpus
# ---------------------------------------------------------------------------
def section_corpus(universe, index, metrics):
    print("\n" + "=" * 72)
    print("1. CORPUS")
    print("=" * 72)

    by_platform = Counter(a["platform"] for a in universe.accounts.values())
    per_entity = Counter()
    for acct in universe.accounts.values():
        per_entity[acct["entity_id"]] += 1
    spread = Counter(per_entity.values())

    print(f"  {len(universe)} accounts over {len(by_platform)} platforms, "
          f"{len(per_entity)} distinct people, {len(index)} registry records")
    for platform, n in sorted(by_platform.items()):
        print(f"    {platform:<12} {n:>6}")
    print("  accounts per person:")
    for k in sorted(spread):
        print(f"    {k} platform(s): {spread[k]:>5} people ({spread[k] / len(per_entity):.1%})")

    print("\n  Field coverage by platform (the missing-data structure the whole")
    print("  system has to work around):")
    header = "    {:<12}" + " {:>10}" * len(IDENTIFYING_FIELDS)
    print(header.format("platform", *[f[:10] for f in IDENTIFYING_FIELDS]))
    coverage = {}
    for platform in sorted(by_platform):
        accts = [a for a in universe.accounts.values() if a["platform"] == platform]
        row = []
        for field in IDENTIFYING_FIELDS:
            present = sum(1 for a in accts if _has(a.get(field)))
            row.append(present / len(accts))
        coverage[platform] = dict(zip(IDENTIFYING_FIELDS, row))
        print(("    {:<12}" + " {:>9.0%}" * len(row)).format(platform, *row))

    metrics["corpus"] = {
        "accounts": len(universe),
        "people": len(per_entity),
        "registry_records": len(index),
        "accounts_per_platform": dict(by_platform),
        "people_by_platform_count": {str(k): v for k, v in sorted(spread.items())},
        "field_coverage": coverage,
    }


def _has(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return str(value).strip() not in ("", "None", "nan")


# ---------------------------------------------------------------------------
# 2. Blocking, with cost
# ---------------------------------------------------------------------------
def section_blocking(universe, vectors, n_true_pairs, metrics):
    print("\n" + "=" * 72)
    print("2. CANDIDATE GENERATION (the ceiling on everything downstream)")
    print("=" * 72)

    t = time.time()
    candidates = blocking.generate_candidates(universe, vectors=vectors, verbose=False)
    elapsed = time.time() - t
    found = int(candidates["label"].sum())
    total_possible = len(universe) * (len(universe) - 1) // 2

    print(f"  {len(candidates)} candidate pairs in {elapsed:.1f}s")
    print(f"  RECALL: {found}/{n_true_pairs} = {found / n_true_pairs:.1%}")
    print(f"  reduction: {len(candidates) / total_possible:.2e} of all "
          f"{total_possible} possible pairs")

    contrib = blocking.path_contributions(candidates, n_true_pairs)
    print("\n  Per-path recall and what only that path found:")
    print(f"    {'path':<16} {'candidates':>11} {'recall':>8} {'unique':>8}")
    for row in contrib.itertuples(index=False):
        print(f"    {row.path:<16} {row.candidates:>11} {row.recall:>7.1%} {row.unique_true_pairs:>8}")

    # What does the union actually lose if a path is removed? "Unique pairs"
    # above overstates each path's value, because a pair found by two paths is
    # unique to neither yet would survive dropping either one.
    print("\n  Leave-one-path-out: the recall and the candidate volume each path")
    print("  is really responsible for.")
    print(f"    {'path removed':<16} {'candidates':>11} {'recall':>8} {'d recall':>9} {'d cost':>9}")
    path_econ = []
    for path in OPTIONAL_PATHS:
        # a pair survives unless EVERY path that produced it was this one
        kept = candidates[candidates["paths"].apply(
            lambda p, path=path: any(x != path for x in p.split(",")))]
        recall = int(kept["label"].sum()) / n_true_pairs
        entry = {
            "path": path,
            "candidates_without": int(len(kept)),
            "recall_without": recall,
            "delta_recall": recall - found / n_true_pairs,
            "delta_candidates": int(len(kept) - len(candidates)),
        }
        path_econ.append(entry)
        print(f"    {path:<16} {entry['candidates_without']:>11} {recall:>7.1%} "
              f"{entry['delta_recall']:>+8.2%} {entry['delta_candidates']:>+9}")

    cheap = [e for e in path_econ if e["delta_recall"] > -0.005]
    if cheap:
        print("\n  Paths whose removal costs under 0.5 points of recall:")
        for e in sorted(cheap, key=lambda e: e["delta_candidates"]):
            print(f"    {e['path']:<16} would save {-e['delta_candidates']:>7} candidates "
                  f"for {-e['delta_recall']:.2%} recall")
        print("  They are kept anyway: candidate volume is not the bottleneck here")
        print("  (scoring 787k pairs takes under a minute), and each of them is the")
        print("  only path that fires on a failure mode the others share - dropping")
        print("  one trades a measured cost against an unmeasured tail risk.")

    metrics["blocking"] = {
        "candidate_pairs": int(len(candidates)),
        "true_pairs": n_true_pairs,
        "recall": found / n_true_pairs,
        "reduction_ratio": len(candidates) / total_possible,
        "seconds": elapsed,
        "per_path": contrib.to_dict("records"),
        "leave_one_out": path_econ,
    }
    return candidates


# ---------------------------------------------------------------------------
# 3. Pairwise scoring + clustering, with a held-out threshold
# ---------------------------------------------------------------------------
def section_resolution(universe, candidates, true_pairs, truth_by_record, metrics):
    print("\n" + "=" * 72)
    print("3. PAIR SCORING AND CLUSTERING")
    print("=" * 72)

    model = FellegiSunterFusion.load(os.path.join(DATA_DIR, "fusion_model.json"))
    pair_list = [(r.record_id_a, r.record_id_b) for r in candidates.itertuples(index=False)]
    t = time.time()
    rows = [extract_pair_features(a, b, universe) for a, b in pair_list]
    scores = model.predict_proba(rows)
    print(f"  scored {len(scores)} pairs in {time.time() - t:.1f}s "
          f"({len(scores) / (time.time() - t):.0f} pairs/s)")

    labels = candidates["label"].to_numpy()

    # Two calibration views, and the gap between them is a finding, not a bug.
    #
    # train_fusion.py splits pairs by entity and evaluates only pairs whose
    # BOTH sides are test entities. That is the right way to avoid leakage,
    # but it silently discards every pair that straddles the split - the
    # majority of all candidates, and all of them negatives, including many of
    # the hardest ones. The deployed system scores every candidate the blocker
    # emits, so the second table is what it actually experiences.
    entities = sorted({a["entity_id"] for a in universe.accounts.values()})
    rng = np.random.default_rng(42)
    order = rng.permutation(len(entities))
    val_entities = {entities[i] for i in order[:len(order) // 3]}
    val_records = {rid for rid, a in universe.accounts.items() if a["entity_id"] in val_entities}
    test_records = set(universe.accounts) - val_records
    all_ids = list(universe.accounts.keys())

    within = np.array([(a in val_records) == (b in val_records)
                       for a, b in pair_list])

    print("\n  Calibration on pairs that stay inside one split "
          "(the protocol train_fusion.py reports):")
    print(f"    {'bin':>10} {'n':>8} {'predicted':>10} {'observed':>10}")
    calib_within = calibration_report(scores[within], labels[within])
    for row in calib_within:
        print(f"    {row['bin']:>10} {row['n']:>8} {row['predicted']:>10.3f} {row['observed']:>10.3f}")
    print(f"    Brier: {brier_score(scores[within], labels[within]):.4f}")

    print("\n  Calibration on EVERY candidate the blocker emits "
          "(what the deployed system sees):")
    print(f"    {'bin':>10} {'n':>8} {'predicted':>10} {'observed':>10}")
    calib = calibration_report(scores, labels)
    for row in calib:
        print(f"    {row['bin']:>10} {row['n']:>8} {row['predicted']:>10.3f} {row['observed']:>10.3f}")
    print(f"    Brier: {brier_score(scores, labels):.4f}")
    print("  The second table is worse, and the reason is structural: an")
    print("  entity-disjoint split throws away the cross-split pairs, which are")
    print("  all negatives and include the hard ones. Any pairwise probability")
    print("  quoted from the first table is therefore optimistic - the numbers")
    print("  to trust downstream are the cluster-level ones in section 4, which")
    print("  are measured on the full pipeline output.")


    print("\n  Configuration comparison (all entities, threshold "
          f"{settings.fusion_match_threshold}):")
    configs = {}
    edges_a = clustering.threshold_edges(pair_list, scores, settings.fusion_match_threshold)
    clusters_a, _ = clustering.build_clusters(edges_a, all_ids)
    edges_b = clustering.constrained_match(pair_list, scores, universe, settings.fusion_match_threshold)
    clusters_b, graph_b = clustering.build_clusters(edges_b, all_ids)
    clusters_c = clustering.enforce_cluster_hygiene(clusters_b, graph_b, universe)

    for name, clusters in (("A) threshold + connected components", clusters_a),
                           ("B) + constrained 1-1 matching", clusters_b),
                           ("C) + cluster hygiene", clusters_c)):
        pw = pairwise_metrics(clusters, truth_by_record, len(true_pairs))
        bc = bcubed(clusters, truth_by_record)
        st = cluster_stats(clusters)
        configs[name[0]] = {"name": name, "pairwise": pw, "bcubed": bc, "stats": st}
        print(f"    {name:<38} pair F1={pw['f1']:.4f}  B3 F1={bc['f1']:.4f}  "
              f"largest={st['largest']}")

    print("\n  Threshold selected on VALIDATION entities, reported on TEST entities.")
    print("  (run_resolution.py sweeps over everything, which tunes on the data it")
    print("   reports; this is the same sweep done honestly.)")
    print(f"    {'thr':>5} {'val B3 F1':>10} {'test B3 F1':>11} {'test pair F1':>13} {'largest':>8}")
    sweep = []
    for thr in [0.30, 0.35, 0.40, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90]:
        edges = clustering.constrained_match(pair_list, scores, universe, thr)
        clusters_t, graph_t = clustering.build_clusters(edges, all_ids)
        clusters_t = clustering.enforce_cluster_hygiene(clusters_t, graph_t, universe)

        # Averaged over held-out accounts only, but scored against the FULL
        # clusters - a cluster that wrongly swallowed a training-set account
        # must still be charged for it.
        val_bc = bcubed(clusters_t, truth_by_record, score_records=val_records)
        test_bc = bcubed(clusters_t, truth_by_record, score_records=test_records)
        test_pairs_n = sum(1 for a, b in true_pairs if a in test_records and b in test_records)
        test_pw = pairwise_metrics([c & test_records for c in clusters_t if c & test_records],
                                   {r: truth_by_record[r] for r in test_records}, test_pairs_n)
        st = cluster_stats(clusters_t)
        sweep.append({"threshold": thr, "val_bcubed_f1": val_bc["f1"],
                      "test_bcubed_f1": test_bc["f1"], "test_pairwise_f1": test_pw["f1"],
                      "test_bcubed_precision": test_bc["precision"],
                      "test_bcubed_recall": test_bc["recall"], "largest": st["largest"]})
        print(f"    {thr:>5.2f} {val_bc['f1']:>10.4f} {test_bc['f1']:>11.4f} "
              f"{test_pw['f1']:>13.4f} {st['largest']:>8}")

    chosen = max(sweep, key=lambda s: s["val_bcubed_f1"])
    best_test = max(sweep, key=lambda s: s["test_bcubed_f1"])
    se = float(np.sqrt(chosen["val_bcubed_f1"] * (1 - chosen["val_bcubed_f1"]) / len(val_records)))
    plateau = [s for s in sweep if s["val_bcubed_f1"] >= chosen["val_bcubed_f1"] - se]

    print(f"\n    Best on validation: threshold {chosen['threshold']:.2f} "
          f"-> TEST B-cubed F1 {chosen['test_bcubed_f1']:.4f} "
          f"(P={chosen['test_bcubed_precision']:.4f} R={chosen['test_bcubed_recall']:.4f}), "
          f"pairwise F1 {chosen['test_pairwise_f1']:.4f}")
    print(f"    Best achievable on test was {best_test['test_bcubed_f1']:.4f} at "
          f"{best_test['threshold']:.2f}; the "
          f"{chosen['test_bcubed_f1'] - best_test['test_bcubed_f1']:+.4f} gap is the "
          "cost of not peeking.")
    print(f"    Standard error on validation B-cubed F1 ~ {se:.4f}, so every threshold from "
          f"{plateau[0]['threshold']:.2f} to {plateau[-1]['threshold']:.2f} is indistinguishable.")
    print("    Picking the argmax inside a plateau this flat is fitting noise. The")
    print("    value that matters is the one downstream, so it is chosen there:")
    print(f"    Deployed value in settings: {settings.fusion_match_threshold}")

    metrics["resolution_threshold_plateau"] = [s["threshold"] for s in plateau]
    metrics["resolution"] = {
        "scoring_seconds": time.time() - t,
        "brier_all_candidates": brier_score(scores, labels),
        "brier_within_split": brier_score(scores[within], labels[within]),
        "calibration_all_candidates": calib,
        "calibration_within_split": calib_within,
        "configs": {k: {"pairwise": v["pairwise"], "bcubed": v["bcubed"], "stats": v["stats"]}
                    for k, v in configs.items()},
        "threshold_sweep": sweep,
        "chosen_threshold": chosen["threshold"],
        "test_at_chosen": chosen,
        "deployed_threshold": settings.fusion_match_threshold,
    }
    return clusters_c, pair_list, scores, [s["threshold"] for s in plateau], val_entities


def section_threshold_downstream(universe, index, pair_list, scores, thresholds,
                                 val_entities, twin_entities, metrics):
    """Choose inside the plateau on the objective that actually matters.

    B-cubed F1 rates a clustering against the truth, but nobody deploys a
    clustering - they deploy an attribution to a real person. Those two
    objectives pull in different directions: a stricter threshold splits
    clusters, which raises cluster purity while starving the mapper of the
    pooled evidence (the LinkedIn job, the Telegram phone) that a cluster
    exists to collect. So the tie inside the plateau is broken here, on
    validation entities, using end-to-end mapping accuracy.
    """
    print("\n  Breaking the plateau tie on end-to-end mapping accuracy")
    print("  (validation entities only):")
    model = FellegiSunterFusion.load(os.path.join(DATA_DIR, "mapper_model.json"))
    with open(os.path.join(DATA_DIR, "mapper_top1_calibration.json"), encoding="utf-8") as f:
        mapper = RealIdentityMapper(index, model, top1_calibration=json.load(f))

    all_ids = list(universe.accounts.keys())
    print(f"    {'thr':>5} {'clusters':>9} {'top-1':>8} {'top-1 twin':>11} {'answered':>9} {'prec|ans':>9}")
    rows = []
    for thr in thresholds:
        edges = clustering.constrained_match(pair_list, scores, universe, thr)
        clusters_t, graph_t = clustering.build_clusters(edges, all_ids)
        clusters_t = clustering.enforce_cluster_hygiene(clusters_t, graph_t, universe)
        val_clusters = [c for c in clusters_t
                        if cluster_true_entity(c, universe) in val_entities]
        candidate_lists = candidates_for_clusters(val_clusters, universe, index)

        correct = twin_correct = twin_n = answered = answered_correct = 0
        for cluster, candidates in zip(val_clusters, candidate_lists):
            truth = cluster_true_entity(cluster, universe)
            ranked = mapper.rank([universe.accounts[r] for r in cluster], candidates, top_n=1)
            hit = bool(ranked and ranked[0]["entity_id"] == truth)
            conf = ranked[0]["confidence_top1"] if ranked else 0.0
            correct += hit
            if truth in twin_entities:
                twin_n += 1
                twin_correct += hit
            if conf >= 0.5:
                answered += 1
                answered_correct += hit

        n = max(1, len(val_clusters))
        row = {"threshold": thr, "n_clusters": len(val_clusters), "top1": correct / n,
               "top1_twin": twin_correct / max(1, twin_n), "answered": answered / n,
               "precision_when_answered": answered_correct / max(1, answered)}
        rows.append(row)
        print(f"    {thr:>5.2f} {len(val_clusters):>9} {row['top1']:>7.1%} "
              f"{row['top1_twin']:>10.1%} {row['answered']:>8.1%} "
              f"{row['precision_when_answered']:>8.1%}")

    best = max(rows, key=lambda r: r["top1"])
    print(f"    -> best end-to-end top-1 at threshold {best['threshold']:.2f} "
          f"({best['top1']:.1%}); deployed value is {settings.fusion_match_threshold}")
    metrics["threshold_downstream"] = rows
    return best


# ---------------------------------------------------------------------------
# 4. Real identity mapping, degradation and error attribution
# ---------------------------------------------------------------------------
def section_mapping(universe, index, pred_clusters, twin_entities, metrics):
    print("\n" + "=" * 72)
    print("4. VIRTUAL -> REAL IDENTITY MAPPING")
    print("=" * 72)

    model = FellegiSunterFusion.load(os.path.join(DATA_DIR, "mapper_model.json"))
    with open(os.path.join(DATA_DIR, "mapper_top1_calibration.json"), encoding="utf-8") as f:
        top1_cal = json.load(f)
    mapper = RealIdentityMapper(index, model, top1_calibration=top1_cal)

    gt_clusters, gt_entities = truth_clusters(universe)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(gt_entities))
    n_fit, n_cal = int(0.5 * len(order)), int(0.2 * len(order))
    split_of = {}
    for pos, idx in enumerate(order):
        split_of[gt_entities[idx]] = "fit" if pos < n_fit else ("cal" if pos < n_fit + n_cal else "test")

    test_pred = [c for c in pred_clusters if split_of.get(cluster_true_entity(c, universe)) == "test"]
    print(f"  {len(test_pred)} predicted clusters belonging to held-out test entities")

    candidate_lists = candidates_for_clusters(test_pred, universe, index)
    records = []
    for cluster, candidates in zip(test_pred, candidate_lists):
        true_entity = cluster_true_entity(cluster, universe)
        members = [universe.accounts[rid] for rid in cluster]
        ranked = mapper.rank(members, candidates, top_n=5)
        conf = (ranked[0]["confidence_top1"] if ranked and ranked[0]["confidence_top1"] is not None
                else (ranked[0]["confidence"] if ranked else 0.0))
        truth_cluster = {rid for rid, a in universe.accounts.items() if a["entity_id"] == true_entity}
        fields = {f for m in members for f in IDENTIFYING_FIELDS if _has(m.get(f))}
        records.append({
            "true_entity": true_entity,
            "predicted": ranked[0]["entity_id"] if ranked else None,
            "top1_correct": bool(ranked and ranked[0]["entity_id"] == true_entity),
            "top5_correct": any(r["entity_id"] == true_entity for r in ranked),
            "confidence": conf,
            "share": ranked[0]["share"] if ranked else 0.0,
            "reachable": true_entity in candidates,
            "twin": true_entity in twin_entities,
            "size": len(cluster),
            "platforms": len({universe.accounts[r]["platform"] for r in cluster}),
            "cluster_pure": all(universe.accounts[r]["entity_id"] == true_entity for r in cluster),
            "cluster_complete": truth_cluster <= cluster,
            "n_fields": len(fields),
            "fields": sorted(fields),
            "n_candidates": len(candidates),
        })
    df = pd.DataFrame(records)

    def block(mask, label):
        sub = df[mask]
        if not len(sub):
            return None
        answered = sub[sub["confidence"] >= 0.5]
        entry = {
            "label": label, "n": int(len(sub)),
            "top1": float(sub["top1_correct"].mean()),
            "top5": float(sub["top5_correct"].mean()),
            "answered": float(len(answered) / len(sub)),
            "precision_when_answered": float(answered["top1_correct"].mean()) if len(answered) else float("nan"),
        }
        print(f"    {label:<24} {entry['n']:>6} {entry['top1']:>7.1%} {entry['top5']:>7.1%} "
              f"{entry['answered']:>9.1%} {entry['precision_when_answered']:>9.1%}")
        return entry

    print(f"\n  End-to-end accuracy on held-out entities")
    print(f"    {'group':<24} {'n':>6} {'top-1':>7} {'top-5':>7} {'answered':>9} {'prec|ans':>9}")
    headline = {
        "all": block(pd.Series(True, index=df.index), "all"),
        "non_twin": block(~df["twin"], "non-twin"),
        "twin": block(df["twin"], "twin"),
    }

    # ---- error attribution ------------------------------------------------
    print("\n  Where the misses come from. Each failed cluster is charged to the")
    print("  earliest stage that could have prevented it, so the counts partition")
    print("  the misses rather than overlapping.")
    misses = df[~df["top1_correct"]]
    unreachable = misses[~misses["reachable"]]
    impure = misses[misses["reachable"] & ~misses["cluster_pure"]]
    incomplete = misses[misses["reachable"] & misses["cluster_pure"] & ~misses["cluster_complete"]]
    ranking = misses[misses["reachable"] & misses["cluster_pure"] & misses["cluster_complete"]]
    taxonomy = {
        "blocking_never_surfaced_the_person": int(len(unreachable)),
        "cluster_contained_someone_else": int(len(impure)),
        "cluster_missing_the_persons_other_accounts": int(len(incomplete)),
        "ranker_had_the_person_and_chose_wrong": int(len(ranking)),
    }
    for label, count in taxonomy.items():
        print(f"    {label:<46} {count:>5}  ({count / max(1, len(df)):.1%} of all clusters)")
    print(f"    {'total misses':<46} {len(misses):>5}")

    # ---- degradation ------------------------------------------------------
    print("\n  Degradation: how much does the system need to know about someone?")
    print(f"    {'identifying fields present':<28} {'n':>6} {'top-1':>8} {'prec|ans':>9}")
    by_fields = []
    for k in sorted(df["n_fields"].unique()):
        sub = df[df["n_fields"] == k]
        answered = sub[sub["confidence"] >= 0.5]
        row = {"n_fields": int(k), "n": int(len(sub)), "top1": float(sub["top1_correct"].mean()),
               "precision_when_answered": float(answered["top1_correct"].mean()) if len(answered) else None}
        by_fields.append(row)
        prec = f"{row['precision_when_answered']:.1%}" if row["precision_when_answered"] is not None else "-"
        print(f"    {k:<28} {row['n']:>6} {row['top1']:>7.1%} {prec:>9}")

    print(f"\n    {'platforms in the cluster':<28} {'n':>6} {'top-1':>8} {'prec|ans':>9}")
    by_platforms = []
    for k in sorted(df["platforms"].unique()):
        sub = df[df["platforms"] == k]
        answered = sub[sub["confidence"] >= 0.5]
        row = {"platforms": int(k), "n": int(len(sub)), "top1": float(sub["top1_correct"].mean()),
               "precision_when_answered": float(answered["top1_correct"].mean()) if len(answered) else None}
        by_platforms.append(row)
        prec = f"{row['precision_when_answered']:.1%}" if row["precision_when_answered"] is not None else "-"
        print(f"    {k:<28} {row['n']:>6} {row['top1']:>7.1%} {prec:>9}")

    print("\n  Which single field, when present, moves the answer most:")
    print(f"    {'field':<20} {'present n':>10} {'top-1 with':>11} {'top-1 without':>14}")
    field_effect = []
    for field in IDENTIFYING_FIELDS:
        with_f = df[df["fields"].apply(lambda fs, f=field: f in fs)]
        without_f = df[df["fields"].apply(lambda fs, f=field: f not in fs)]
        if not len(with_f) or not len(without_f):
            continue
        row = {"field": field, "n_with": int(len(with_f)),
               "top1_with": float(with_f["top1_correct"].mean()),
               "top1_without": float(without_f["top1_correct"].mean())}
        field_effect.append(row)
        print(f"    {field:<20} {row['n_with']:>10} {row['top1_with']:>10.1%} "
              f"{row['top1_without']:>13.1%}")
    print("  Read this as association, not causation: LinkedIn supplies job,")
    print("  education and city together, so their columns largely describe the")
    print("  same population - people whose cluster reached LinkedIn at all.")

    # ---- calibration ------------------------------------------------------
    print("\n  Displayed confidence vs. reality on held-out clusters:")
    print(f"    {'bin':>10} {'n':>7} {'predicted':>10} {'observed':>10}")
    calib = calibration_report(df["confidence"].to_numpy(),
                               df["top1_correct"].to_numpy().astype(int))
    for row in calib:
        print(f"    {row['bin']:>10} {row['n']:>7} {row['predicted']:>10.3f} {row['observed']:>10.3f}")
    print(f"    Brier score: {brier_score(df['confidence'].to_numpy(), df['top1_correct'].to_numpy().astype(int)):.4f}")

    print("\n  Contested attributions (runner-up within 10% of the leader):")
    contested = df[df["share"] < 0.55]
    print(f"    {len(contested)} of {len(df)} clusters ({len(contested) / len(df):.1%}); "
          f"top-1 accuracy there is {contested['top1_correct'].mean():.1%} "
          f"versus {df[df['share'] >= 0.55]['top1_correct'].mean():.1%} elsewhere.")
    print(f"    Twins are {contested['twin'].mean():.1%} of them, versus "
          f"{df['twin'].mean():.1%} of all clusters - which is exactly what the")
    print("    share statistic is supposed to detect.")

    metrics["mapping"] = {
        "headline": headline,
        "error_taxonomy": taxonomy,
        "total_misses": int(len(misses)),
        "n_clusters_evaluated": int(len(df)),
        "by_field_count": by_fields,
        "by_platform_count": by_platforms,
        "field_effect": field_effect,
        "calibration": calib,
        "brier": brier_score(df["confidence"].to_numpy(), df["top1_correct"].to_numpy().astype(int)),
        "contested": {
            "n": int(len(contested)),
            "share_of_clusters": float(len(contested) / len(df)),
            "top1_contested": float(contested["top1_correct"].mean()),
            "top1_uncontested": float(df[df["share"] >= 0.55]["top1_correct"].mean()),
            "twin_rate_contested": float(contested["twin"].mean()),
            "twin_rate_overall": float(df["twin"].mean()),
        },
    }
    return df


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    metrics = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    print("Loading models and data...")
    client = QdrantClient(settings.qdrant_url, timeout=120.0)
    vectors = text_model.load_vectors_from_qdrant(client, COLLECTION_NAME)
    universe = AccountUniverse.load(DATA_DIR, vectors=vectors)
    index = RegistryIndex.load(DATA_DIR)

    gt = pd.read_parquet(os.path.join(DATA_DIR, "ground_truth_pairs.parquet"))
    true_pairs = {tuple(sorted((r.record_id_a, r.record_id_b))) for r in gt.itertuples(index=False)}
    truth_by_record = {rid: a["entity_id"] for rid, a in universe.accounts.items()}

    registry_df = pd.read_parquet(os.path.join(DATA_DIR, "real_identity_registry.parquet"))
    twin_entities = set(registry_df[registry_df["twin_of"].notna()]["entity_id"])
    twin_entities |= set(registry_df[registry_df["twin_of"].notna()]["twin_of"].dropna())

    section_corpus(universe, index, metrics)
    candidates = section_blocking(universe, vectors, len(true_pairs), metrics)
    (pred_clusters, pair_list, scores, plateau,
     val_entities) = section_resolution(universe, candidates, true_pairs, truth_by_record, metrics)
    section_threshold_downstream(universe, index, pair_list, scores, plateau,
                                 val_entities, twin_entities, metrics)
    section_mapping(universe, index, pred_clusters, twin_entities, metrics)

    real_path = os.path.join(DATA_DIR, "real_validation", "metrics.json")
    if os.path.exists(real_path):
        with open(real_path, encoding="utf-8") as f:
            metrics["real_data_validation"] = json.load(f)
        print("\n  (real-data component validation folded in from "
              "data/real_validation/metrics.json)")

    metrics["total_seconds"] = time.time() - t0
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nWrote {METRICS_PATH}  (total runtime {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
