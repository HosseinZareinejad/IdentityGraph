"""
Phase 6 - choose which modalities the fusion model should actually keep.

The leave-one-out ablation in train_fusion.py produced an uncomfortable result:
removing stylometry (+0.009 AP) or graph (+0.010 AP) makes the model BETTER,
and metadata, text and temporal are worth roughly nothing on their own. Only
the name modality carries real signal (AP 0.876 alone, versus 0.894 for the
full model).

That is not noise, it is Fellegi-Sunter's conditional-independence assumption
being violated. Each weak feature contributes its own log-odds term as though
it were fresh evidence; when a feature is mostly noise, it adds variance to
every score without adding information, and when several weak features are
correlated with each other it adds that variance several times over.

So rather than shipping every modality because the proposal listed five
models, this script measures every subset that matters and picks one - ON THE
VALIDATION SPLIT, never on test. The chosen set is written to
data/feature_set.json and consumed by the training and serving paths.

The honest counter-argument to naive subset selection is kept in view: average
precision is dominated by the common case, while phone_exact and email_exact
are rare and decisive. A set that wins on AP while throwing away the only
features that can settle a twin is not actually better, so the report below
prints both AP and the recall on pairs that carry exact contact evidence.
"""
import itertools
import json
import os
import sys
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src import blocking  # noqa: E402
from src.config import settings  # noqa: E402
from src.models import text_model  # noqa: E402
from src.models.fusion_model import FellegiSunterFusion  # noqa: E402
from src.models.pair_features import (MODALITY_FEATURES, AccountUniverse,  # noqa: E402
                                      extract_pair_features)
from scripts.train_fusion import split_by_entity  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
COLLECTION_NAME = "identity_accounts"
OUT_PATH = os.path.join(DATA_DIR, "feature_set.json")

# "name" is never a candidate for removal - it is the only modality with
# standalone discriminative power, so every subset contains it.
OPTIONAL = ["metadata", "text", "stylometry", "temporal", "graph"]


def subset_rows(rows, keep_features):
    return [{k: v for k, v in row.items() if k in keep_features} for row in rows]


def score_subset(train_rows, train_labels, val_rows, val_labels, keep_features, contact_mask):
    model = FellegiSunterFusion().fit(subset_rows(train_rows, keep_features), train_labels)
    probs = model.predict_proba(subset_rows(val_rows, keep_features))
    labels = np.asarray(val_labels)

    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 1.0, 0.05):
        pred = probs >= thr
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    # Recall restricted to the true pairs that carry exact phone/email
    # evidence - the rare cases where a dropped modality would be catastrophic
    # rather than merely slightly worse on average.
    decisive = contact_mask & (labels == 1)
    decisive_recall = float((probs[decisive] >= best_thr).mean()) if decisive.any() else float("nan")

    return {
        "auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
        "f1": best_f1,
        "threshold": best_thr,
        "decisive_recall": decisive_recall,
        "n_features": len(keep_features),
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()

    print("Loading universe...")
    client = QdrantClient(settings.qdrant_url, timeout=120.0)
    vectors = text_model.load_vectors_from_qdrant(client, COLLECTION_NAME)
    universe = AccountUniverse.load(DATA_DIR, vectors=vectors)

    print("Blocking + feature extraction...")
    pairs = blocking.generate_candidates(universe, vectors=vectors, verbose=False).reset_index(drop=True)
    rows = [extract_pair_features(r.record_id_a, r.record_id_b, universe)
            for r in pairs.itertuples(index=False)]
    print(f"  {len(pairs)} candidate pairs ({int(pairs['label'].sum())} true)")

    train_pairs, val_pairs, _ = split_by_entity(pairs)
    train_rows = [rows[i] for i in train_pairs.index]
    val_rows = [rows[i] for i in val_pairs.index]
    train_labels = train_pairs["label"].tolist()
    val_labels = val_pairs["label"].tolist()
    contact_mask = np.array([bool(r.get("phone_exact") or r.get("email_exact")) for r in val_rows])
    print(f"  fit={len(train_rows)}  validation={len(val_rows)} "
          f"({int(contact_mask.sum())} of them carry exact contact evidence)")

    all_optional = set(OPTIONAL)
    candidates = []
    for size in range(len(OPTIONAL) + 1):
        for combo in itertools.combinations(OPTIONAL, size):
            candidates.append(set(combo))

    print(f"\nEvaluating {len(candidates)} modality subsets on the VALIDATION split")
    print(f"  {'modalities kept':<46} {'AUC':>7} {'AP':>7} {'F1':>7} {'thr':>5} {'decisive R':>11}")
    results = []
    for optional_kept in candidates:
        modalities = ["name"] + [m for m in OPTIONAL if m in optional_kept]
        keep = {f for m in modalities for f in MODALITY_FEATURES[m]}
        res = score_subset(train_rows, train_labels, val_rows, val_labels, keep, contact_mask)
        res["modalities"] = modalities
        results.append(res)
        label = "+".join(modalities)
        print(f"  {label:<46} {res['auc']:>7.4f} {res['ap']:>7.4f} {res['f1']:>7.4f} "
              f"{res['threshold']:>5.2f} {res['decisive_recall']:>10.1%}")

    results.sort(key=lambda r: -r["f1"])
    print("\n  Top 5 by validation F1:")
    for res in results[:5]:
        print(f"    {'+'.join(res['modalities']):<46} F1={res['f1']:.4f} AP={res['ap']:.4f} "
              f"decisiveR={res['decisive_recall']:.1%}")

    best = results[0]
    full = next(r for r in results if len(r["modalities"]) == len(OPTIONAL) + 1)
    name_only = next(r for r in results if r["modalities"] == ["name"])

    # How large a difference is even real? With this many validation pairs the
    # standard error on F1 is around one half-point, so anything smaller is
    # sampling noise and selecting on it would just be fitting the validation
    # split. Approximated from the positive count, which is what F1 rests on.
    n_pos = int(np.sum(val_labels))
    se = float(np.sqrt(best["f1"] * (1 - best["f1"]) / n_pos))
    margin = max(2 * se, 0.005)
    print(f"\n  Spread across all {len(results)} subsets: "
          f"{results[-1]['f1']:.4f} - {results[0]['f1']:.4f}")
    print(f"  Standard error on validation F1 ~ {se:.4f} ({n_pos} positives); "
          f"requiring a {margin:.4f} margin to prefer a subset over the full model")

    if best["f1"] - full["f1"] < margin:
        print("  -> the best subset does NOT beat the full model by more than noise.")
        print("     Keeping every modality: the extra features cost nothing at serving")
        print("     time, and dropping one on a difference this size would be fitting")
        print("     the validation split rather than learning anything about the data.")
        best = full
    elif best["decisive_recall"] < full["decisive_recall"] - 0.01:
        # Never trade away the ability to close a case on exact contact
        # evidence for a fractional average-case gain.
        print("  -> best-F1 subset loses recall on decisive contact evidence; "
              "falling back to the best subset that does not.")
        best = next(r for r in results if r["decisive_recall"] >= full["decisive_recall"] - 0.01)

    chosen = {f for m in best["modalities"] for f in MODALITY_FEATURES[m]}
    payload = {
        "modalities": best["modalities"],
        "features": sorted(chosen),
        "selected_on": "validation split (entity-disjoint); a subset is preferred only if it "
                       "beats the full model by more than two standard errors of F1",
        "validation": {k: best[k] for k in ("auc", "ap", "f1", "threshold", "decisive_recall")},
        "full_model_validation": {k: full[k] for k in ("auc", "ap", "f1", "threshold", "decisive_recall")},
        "name_only_validation": {k: name_only[k] for k in ("auc", "ap", "f1", "threshold", "decisive_recall")},
        "all_subsets": [{"modalities": r["modalities"], "f1": r["f1"], "ap": r["ap"],
                         "decisive_recall": r["decisive_recall"]} for r in results],
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n  CHOSEN: {'+'.join(best['modalities'])}")
    print(f"  Name alone reaches F1 {name_only['f1']:.4f}, i.e. "
          f"{100 * name_only['f1'] / full['f1']:.1f}% of the full model. The other five")
    print("  modalities together are worth "
          f"{full['f1'] - name_only['f1']:+.4f} F1 - real, but small. The one place")
    print("  they are not optional is decisive contact evidence, where metadata takes")
    print(f"  recall from {name_only['decisive_recall']:.1%} to {full['decisive_recall']:.1%}.")
    print(f"  -> {OUT_PATH}")
    print(f"\nTotal runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
