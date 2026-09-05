"""
Phase 3 - build the training pair set, fit the fusion model, evaluate, ablate.

Two methodological points that decide whether the resulting numbers mean
anything:

1. NEGATIVES COME FROM THE CANDIDATE DISTRIBUTION, NOT AT RANDOM.
   Sampling random non-matching pairs would make the task trivial (two random
   people share nothing) and the learned weights and calibration would be
   useless in production, where the scorer only ever sees pairs that blocking
   surfaced - i.e. pairs that already look similar. So negatives are drawn
   from the top-K most name-similar accounts on the other platform, which is
   what a real blocking stage would hand over. The twin hard negatives
   (identical name + city + birth year, different people) land in this set
   automatically.

2. THE TRAIN/TEST SPLIT IS BY ENTITY, NOT BY PAIR.
   A person with accounts on 3 platforms produces 3 pairs. Splitting by pair
   would put one of their pairs in train and another in test, leaking that
   person's names, style and posting rhythm across the split. Splitting by
   entity_id keeps every pair of a given person on one side.
"""
import os
import sys
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from rapidfuzz import process as rf_process
from rapidfuzz.distance import JaroWinkler
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.config import settings  # noqa: E402
from src.models import text_model  # noqa: E402
from src.models.fusion_model import (FellegiSunterFusion, brier_score,  # noqa: E402
                                     calibration_report)
from src.models.pair_features import (MODALITY_FEATURES, AccountUniverse,  # noqa: E402
                                      PLATFORMS, extract_pair_features)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "data", "fusion_model.json")
COLLECTION_NAME = "identity_accounts"

CANDIDATES_PER_ACCOUNT = 10   # top-K name-similar accounts per (source, target platform)
TEST_ENTITY_FRACTION = 0.3
CALIBRATION_ENTITY_FRACTION = 0.15
RANDOM_SEED = 42


def build_candidate_pairs(universe: AccountUniverse) -> pd.DataFrame:
    """Blocking stand-in: for every ordered platform pair, take each account's
    top-K most name-similar accounts on the other platform.

    Phase 4 replaces this with a properly tuned multi-path blocker; it exists
    here so the fusion model trains on a realistic candidate distribution.
    """
    by_platform = {p: [] for p in PLATFORMS}
    for rid, acct in universe.accounts.items():
        by_platform[acct["platform"]].append(rid)

    pairs = set()
    for i, plat_a in enumerate(PLATFORMS):
        for plat_b in PLATFORMS[i + 1:]:
            ids_a, ids_b = by_platform[plat_a], by_platform[plat_b]
            names_a = [str(universe.accounts[r]["display_name"] or "") for r in ids_a]
            names_b = [str(universe.accounts[r]["display_name"] or "") for r in ids_b]

            # C-optimised all-pairs similarity, then top-K per row
            sim = rf_process.cdist(names_a, names_b, scorer=JaroWinkler.normalized_similarity,
                                   workers=-1, dtype=np.float32)
            top_k = np.argpartition(-sim, kth=min(CANDIDATES_PER_ACCOUNT, sim.shape[1] - 1),
                                    axis=1)[:, :CANDIDATES_PER_ACCOUNT]
            for row_idx, col_indices in enumerate(top_k):
                for col_idx in col_indices:
                    pairs.add((ids_a[row_idx], ids_b[col_idx]))
            print(f"    {plat_a}-{plat_b}: {sim.shape[0]}x{sim.shape[1]} -> {len(pairs)} cumulative pairs")

    rows = []
    for rid_a, rid_b in pairs:
        a, b = universe.accounts[rid_a], universe.accounts[rid_b]
        rows.append({
            "record_id_a": rid_a, "record_id_b": rid_b,
            "entity_id_a": a["entity_id"], "entity_id_b": b["entity_id"],
            "label": int(a["entity_id"] == b["entity_id"]),
        })
    return pd.DataFrame(rows)


def split_by_entity(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three disjoint entity groups: weight-fitting, calibration, test.

    The calibration set must not overlap the weight-fitting set, otherwise
    isotonic regression fits the scores the model already memorised and
    reports calibration that won't hold on new data.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    # pd.unique returns a pandas ArrowStringArray here; numpy's shuffle warns
    # that it can duplicate elements on non-Sequence array types (which would
    # silently break split disjointness and leak entities across folds).
    # Verified disjoint in practice, but convert to a plain object array so
    # correctness doesn't rest on unspecified behaviour.
    entities = np.asarray(pd.unique(pd.concat([pairs["entity_id_a"], pairs["entity_id_b"]])), dtype=object)
    rng.shuffle(entities)

    n_test = int(len(entities) * TEST_ENTITY_FRACTION)
    n_calib = int(len(entities) * CALIBRATION_ENTITY_FRACTION)
    test_entities = set(entities[:n_test])
    calib_entities = set(entities[n_test:n_test + n_calib])

    def both_in(group):
        return pairs["entity_id_a"].isin(group) & pairs["entity_id_b"].isin(group)

    in_test = both_in(test_entities)
    in_calib = both_in(calib_entities)
    # A pair belongs to the fit set only if neither side touches the other groups
    touches_other = (pairs["entity_id_a"].isin(test_entities | calib_entities)
                     | pairs["entity_id_b"].isin(test_entities | calib_entities))
    return pairs[~touches_other].copy(), pairs[in_calib].copy(), pairs[in_test].copy()


def evaluate(model, rows, labels, threshold_sweep=True) -> dict:
    probs = model.predict_proba(rows)
    labels = np.asarray(labels)
    result = {
        "auc": roc_auc_score(labels, probs) if labels.sum() else float("nan"),
        "average_precision": average_precision_score(labels, probs) if labels.sum() else float("nan"),
    }
    if threshold_sweep:
        best = None
        for thr in np.arange(0.05, 1.0, 0.05):
            pred = probs >= thr
            tp = int((pred & (labels == 1)).sum())
            fp = int((pred & (labels == 0)).sum())
            fn = int((~pred & (labels == 1)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if best is None or f1 > best["f1"]:
                best = {"threshold": float(thr), "precision": precision, "recall": recall, "f1": f1}
        result["best_f1"] = best
    return result


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    t0 = time.time()
    print("Loading vectors from Qdrant...")
    client = QdrantClient(settings.qdrant_url, timeout=120.0)
    vectors = text_model.load_vectors_from_qdrant(client, COLLECTION_NAME)
    print(f"  {len(vectors)} vectors")

    print("Loading account universe (accounts, style profiles, hour profiles)...")
    universe = AccountUniverse.load(DATA_DIR, vectors=vectors)
    print(f"  {len(universe)} accounts")

    print("Building candidate pairs (blocking stand-in)...")
    pairs = build_candidate_pairs(universe)
    print(f"  {len(pairs)} candidate pairs, {int(pairs['label'].sum())} of them true matches")

    gt = pd.read_parquet(os.path.join(DATA_DIR, "ground_truth_pairs.parquet"))
    print(f"  blocking recall: {pairs['label'].sum()}/{len(gt)} = {pairs['label'].sum()/len(gt):.1%} "
          f"of all true cross-platform pairs")

    print("Extracting pair features...")
    t = time.time()
    feature_rows = [extract_pair_features(r.record_id_a, r.record_id_b, universe)
                    for r in pairs.itertuples(index=False)]
    print(f"  done in {time.time() - t:.1f}s")

    pairs = pairs.reset_index(drop=True)
    train_pairs, calib_pairs, test_pairs = split_by_entity(pairs)
    train_rows = [feature_rows[i] for i in train_pairs.index]
    calib_rows = [feature_rows[i] for i in calib_pairs.index]
    test_rows = [feature_rows[i] for i in test_pairs.index]
    test_labels = test_pairs["label"].to_numpy()
    print(f"\nSplit by entity (disjoint): fit={len(train_pairs)} pairs ({int(train_pairs['label'].sum())} pos), "
          f"calibration={len(calib_pairs)} ({int(calib_pairs['label'].sum())} pos), "
          f"test={len(test_pairs)} ({int(test_pairs['label'].sum())} pos)")

    print("\nFitting Fellegi-Sunter fusion model...")
    model = FellegiSunterFusion().fit(train_rows, train_pairs["label"].tolist())

    print("Calibration BEFORE isotonic fit (predicted vs. observed):")
    raw_probs = model.predict_proba(test_rows)
    for row in calibration_report(raw_probs, test_labels):
        print(f"  {row['bin']}: n={row['n']:6d}  predicted={row['predicted']:.3f}  observed={row['observed']:.3f}")
    print(f"  Brier score (lower is better): {brier_score(raw_probs, test_labels):.4f}")

    model.fit_calibration(calib_rows, calib_pairs["label"].tolist())
    model.save(MODEL_PATH)
    print(f"\n  saved to {MODEL_PATH}")

    print("\n=== Test-set performance (full model, calibrated) ===")
    full = evaluate(model, test_rows, test_pairs["label"].tolist())
    print(f"  ROC-AUC: {full['auc']:.4f}   Average precision: {full['average_precision']:.4f}")
    b = full["best_f1"]
    print(f"  Best F1: {b['f1']:.4f} at threshold {b['threshold']:.2f} "
          f"(precision {b['precision']:.4f}, recall {b['recall']:.4f})")

    print("\n=== Calibration AFTER isotonic fit ===")
    cal_probs = model.predict_proba(test_rows)
    for row in calibration_report(cal_probs, test_labels):
        print(f"  {row['bin']}: n={row['n']:6d}  predicted={row['predicted']:.3f}  observed={row['observed']:.3f}")
    print(f"  Brier score (lower is better): {brier_score(cal_probs, test_labels):.4f}")

    print("\n=== Ablation: leave-one-modality-out (test AUC / AP / F1) ===")
    print(f"  {'full model':<22} AUC={full['auc']:.4f}  AP={full['average_precision']:.4f}  F1={b['f1']:.4f}")
    for modality, feats in MODALITY_FEATURES.items():
        ablated_train = [{k: v for k, v in row.items() if k not in feats} for row in train_rows]
        ablated_test = [{k: v for k, v in row.items() if k not in feats} for row in test_rows]
        m = FellegiSunterFusion().fit(ablated_train, train_pairs["label"].tolist())
        res = evaluate(m, ablated_test, test_pairs["label"].tolist())
        print(f"  {'without ' + modality:<22} AUC={res['auc']:.4f}  AP={res['average_precision']:.4f}  "
              f"F1={res['best_f1']['f1']:.4f}   (dAP={res['average_precision'] - full['average_precision']:+.4f})")

    print("\n=== Modality alone (test AP) ===")
    for modality, feats in MODALITY_FEATURES.items():
        only_train = [{k: v for k, v in row.items() if k in feats} for row in train_rows]
        only_test = [{k: v for k, v in row.items() if k in feats} for row in test_rows]
        m = FellegiSunterFusion().fit(only_train, train_pairs["label"].tolist())
        res = evaluate(m, only_test, test_pairs["label"].tolist(), threshold_sweep=False)
        print(f"  {modality + ' only':<22} AUC={res['auc']:.4f}  AP={res['average_precision']:.4f}")

    print("\n=== Strongest learned evidence weights (log2 m/u) ===")
    flat = [(f, lvl, w) for f, levels in model.weights.items() for lvl, w in levels.items() if w != 0.0]
    flat.sort(key=lambda t: abs(t[2]), reverse=True)
    for feature, level, weight in flat[:15]:
        print(f"  {feature:28s} {level:>8s}  {weight:+.2f} bits")

    print(f"\nTotal runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
