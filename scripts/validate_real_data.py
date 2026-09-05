"""
Phase 6 - validate the name components against REAL Persian names.

Every number in Phases 1-5 is measured on a synthetic world, which leaves one
obvious objection open: the noise model was written by the same person who
wrote the matcher, so the matcher may only be good at beating its own author's
idea of noise. This script closes part of that gap by testing the name
machinery on ~7.8k real (Persian label, English label) pairs of Iranian people
harvested from Wikidata - romanisations produced by many different human
editors, with no involvement from this project.

It cannot validate the whole system (there is no public labelled cross-platform
account corpus for Persian), so it deliberately validates only what real data
CAN speak to, and says so:

  1. TRANSLITERATION FIDELITY - how close does the deterministic fa->latin map
     land to a human romanisation? This underwrites the `cross_field` blocking
     path and the `username_vs_name` feature.

  2. CROSS-SCRIPT RETRIEVAL - the actual task: given a Persian name, rank 7.8k
     real English names. Top-K accuracy here is a real-data measurement of the
     path that links a Latin handle to a Persian legal name.

  3. BLOCKING-KEY DISCRIMINATIVENESS UNDER A REAL NAME DISTRIBUTION - real
     Persian FIRST names are heavily Zipfian (محمد, علی, فاطمه dominate),
     which predicts large, expensive phonetic blocks. The measurement says
     otherwise, and the prediction was wrong: keyed on the FULL name, 99.7% of
     real blocks are singletons. Surnames carry the entropy that first names
     lack. Reported anyway, because the cost of this path is exactly what the
     MAX_BLOCK_SIZE cutoff exists to bound.

  4. HOMOPHONE COLLAPSING ON REAL SPELLING VARIANTS - does the phonetic key
     merge the ز/ذ, س/ص/ث, ت/ط, ق/غ variants that real Persian actually
     produces, and how much does it over-merge while doing so?
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from rapidfuzz import process as rf_process
from rapidfuzz.distance import JaroWinkler

from src.blocking import MAX_BLOCK_SIZE
from src.models import name_model
from src.text_utils import compute_phonetic_key, latin_skeleton, transliterate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS_PATH = os.path.join(BASE_DIR, "data", "real_validation", "name_pairs.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "real_validation", "metrics.json")

PERSIAN_RANGE = "؀-ۿ"


def normalize_latin(text: str) -> str:
    """Strip everything two romanisations can legitimately disagree about:
    case, spaces, hyphens, apostrophes. What remains is the consonant/vowel
    skeleton, which is what the transliterator is actually trying to predict."""
    return re.sub(r"[^a-z]", "", (text or "").lower())


def load_pairs() -> pd.DataFrame:
    df = pd.read_csv(PAIRS_PATH)
    df = df.dropna(subset=["fa_label", "en_label"])
    df["fa_label"] = df["fa_label"].astype(str).str.strip()
    df["en_label"] = df["en_label"].astype(str).str.strip()
    # Wikidata editors sometimes mirror the Persian label into the English
    # field. Keeping those would flatter the transliterator, so require Latin
    # letters on one side and Persian letters on the other.
    df = df[df["en_label"].str.contains("[A-Za-z]", regex=True)]
    df = df[df["fa_label"].str.contains("[" + PERSIAN_RANGE + "]", regex=True)]
    return df.drop_duplicates(subset=["fa_label", "en_label"]).reset_index(drop=True)


def section_transliteration(df: pd.DataFrame, results: dict) -> np.ndarray:
    print("\n=== 1. Transliteration fidelity vs. human romanisation ===")
    predicted = [normalize_latin(transliterate(name_model.strip_titles(n))) for n in df["fa_label"]]
    actual = [normalize_latin(n) for n in df["en_label"]]

    sims = np.array([JaroWinkler.normalized_similarity(p, a) for p, a in zip(predicted, actual)])
    exact = float(np.mean([p == a for p, a in zip(predicted, actual)]))

    skel_p = [latin_skeleton(p) for p in predicted]
    skel_a = [latin_skeleton(a) for a in actual]
    skel_sims = np.array([JaroWinkler.normalized_similarity(p, a)
                          for p, a in zip(skel_p, skel_a)])

    print("  n = %d real name pairs" % len(df))
    print("  exact skeleton match : %.1f%%" % (100 * exact))
    print("  Jaro-Winkler mean    : %.3f   median %.3f" % (sims.mean(), np.median(sims)))
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        print("    p%-3d %.3f" % (int(q * 100), np.quantile(sims, q)))
    for thr in (0.70, 0.80, 0.85, 0.90):
        print("  share with JW >= %.2f   : %.1f%%" % (thr, 100 * float((sims >= thr).mean())))
    print("  comparing CONSONANT SKELETONS instead (what the matcher does):")
    print("    Jaro-Winkler mean  : %.3f   exact match %.1f%%"
          % (skel_sims.mean(), 100 * np.mean([p == a for p, a in zip(skel_p, skel_a)])))
    print("    share with JW >= 0.85: %.1f%%" % (100 * float((skel_sims >= 0.85).mean())))
    print("  The gap between those two lines is the short vowels, which Persian")
    print("  script does not record and the transliterator therefore cannot know.")

    # The failures are the informative part: which sounds does the map get
    # wrong? Worth showing rather than averaging away.
    print("  worst-scoring examples (fa -> ours | human):")
    for i in np.argsort(sims)[:8]:
        print("    %-28s %-22s | %-22s %.2f"
              % (df["fa_label"].iloc[i], predicted[i], actual[i], sims[i]))

    results["transliteration"] = {
        "n": int(len(df)),
        "exact_skeleton_match": exact,
        "jw_mean": float(sims.mean()),
        "jw_median": float(np.median(sims)),
        "share_jw_ge_0.80": float((sims >= 0.80).mean()),
        "share_jw_ge_0.85": float((sims >= 0.85).mean()),
        "skeleton_jw_mean": float(skel_sims.mean()),
        "skeleton_exact_match": float(np.mean([p == a for p, a in zip(skel_p, skel_a)])),
        "skeleton_share_jw_ge_0.85": float((skel_sims >= 0.85).mean()),
    }
    return sims


def section_retrieval(df: pd.DataFrame, results: dict):
    """The real task: Persian name in, ranked English names out."""
    print("\n=== 2. Cross-script retrieval over the full real corpus ===")
    print("  ranking each Persian name against all %d real English names" % len(df))

    base_q = [transliterate(name_model.strip_titles(n)) for n in df["fa_label"]]
    base_c = list(df["en_label"])

    row = {"corpus_size": int(len(df))}
    for space, fn in (("literal", normalize_latin),
                      ("consonant_skeleton", latin_skeleton)):
        queries = [fn(q) for q in base_q]
        corpus = [fn(c) for c in base_c]
        sim = rf_process.cdist(queries, corpus, scorer=JaroWinkler.normalized_similarity,
                               workers=-1, dtype=np.float32)
        truth_scores = sim[np.arange(len(sim)), np.arange(len(sim))]
        # Rank of the correct answer, counting strictly-better candidates so
        # ties are scored optimistically-but-consistently.
        ranks = (sim > truth_scores[:, None]).sum(axis=1) + 1

        entry = {}
        for k in (1, 5, 10, 20, 50):
            entry["top%d" % k] = float((ranks <= k).mean())
        entry["median_rank"] = int(np.median(ranks))
        entry["mrr"] = float(np.mean(1.0 / ranks))
        row[space] = entry
        print("  %-20s top-1 %.1f%%  top-5 %.1f%%  top-10 %.1f%%  top-50 %.1f%%  MRR %.3f"
              % (space, 100 * entry["top1"], 100 * entry["top5"], 100 * entry["top10"],
                 100 * entry["top50"], entry["mrr"]))

    print("  The second row is what the pipeline uses. Persian never wrote the")
    print("  short vowels down, so two romanisations of one name disagree almost")
    print("  only about vowels; dropping them from both sides removes the one")
    print("  thing neither side could know.")
    print("  NOTE: a 1-in-%d retrieval with NO blocking, no city, no birth year" % len(df))
    print("  and no other evidence - the hardest possible framing of the name")
    print("  path alone. In the pipeline it only has to place the right person")
    print("  in a top-10 candidate list that the other features then re-rank.")

    results["cross_script_retrieval"] = row


def section_blocking_distribution(df: pd.DataFrame, results: dict):
    print("\n=== 3. Blocking-key behaviour under a REAL name distribution ===")
    sizes = Counter(compute_phonetic_key(n) for n in df["fa_label"])
    size_values = np.array(sorted(sizes.values(), reverse=True))

    n = len(df)
    singleton_share = float((size_values == 1).sum() / len(size_values))
    pairs = int(sum(int(s) * (int(s) - 1) // 2 for s in size_values))
    all_pairs = n * (n - 1) // 2
    over_cap = int((size_values > MAX_BLOCK_SIZE).sum())
    lost = int(size_values[size_values > MAX_BLOCK_SIZE].sum())

    print("  %d distinct phonetic keys over %d real names" % (len(size_values), n))
    print("  singleton keys       : %.1f%%" % (100 * singleton_share))
    print("  largest block        : %d names" % size_values[0])
    print("  top-5 block sizes    : %s" % [int(v) for v in size_values[:5]])
    print("  pairs generated      : %d = %.2e of all pairs" % (pairs, pairs / all_pairs))
    print("  blocks over MAX_BLOCK_SIZE=%d: %d (covering %d names, %.1f%%)"
          % (MAX_BLOCK_SIZE, over_cap, lost, 100 * lost / n))
    print("  -> the Zipfian-first-name worry does not survive contact with the")
    print("     data: keyed on the full name, real blocks are almost all")
    print("     singletons, and the cap never fires. The key is cheap and highly")
    print("     discriminative on real names - its weakness is recall, not cost,")
    print("     which is why the pipeline pairs it with fuzzy paths.")

    results["phonetic_blocking_real"] = {
        "n_names": n,
        "distinct_keys": int(len(size_values)),
        "singleton_key_share": singleton_share,
        "largest_block": int(size_values[0]),
        "pairs_generated": pairs,
        "reduction_ratio": pairs / all_pairs,
        "blocks_over_cap": over_cap,
        "names_in_over_cap_blocks_share": lost / n,
    }


def section_homophones(df: pd.DataFrame, results: dict):
    print("\n=== 4. Homophone collapsing on real spelling variants ===")
    substitutions = [("ذ", "ز"), ("ص", "س"), ("ث", "س"),
                     ("ط", "ت"), ("غ", "ق"), ("ض", "ز")]

    merged, checked = 0, 0
    examples = []
    for name in df["fa_label"]:
        variant, applied = name, False
        for a, b in substitutions:
            if a in variant:
                variant = variant.replace(a, b)
                applied = True
        if not applied:
            continue
        checked += 1
        if compute_phonetic_key(name) == compute_phonetic_key(variant):
            merged += 1
            if len(examples) < 5:
                examples.append((name, variant))

    print("  %d of %d real names contain a homophone-ambiguous letter" % (checked, len(df)))
    print("  key survives the substitution: %d/%d = %.1f%%"
          % (merged, checked, 100 * merged / max(1, checked)))
    for original, variant in examples:
        print("    %s  ==  %s" % (original, variant))

    # The other half of the question: how much does it over-merge? Distinct
    # names collapsing onto one key is the price paid for the recall above.
    naive = int(df["fa_label"].nunique())
    keyed = int(df["fa_label"].map(compute_phonetic_key).nunique())
    print("  %d distinct names -> %d distinct keys (%.1f%% collapse rate)"
          % (naive, keyed, 100 * (1 - keyed / naive)))

    results["homophone_collapsing"] = {
        "names_with_ambiguous_letter": checked,
        "variant_merge_rate": merged / max(1, checked),
        "distinct_names": naive,
        "distinct_keys": keyed,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = load_pairs()
    print("Phase 6 - real-data component validation")
    print("Source: %s (%d usable pairs after filtering)" % (PAIRS_PATH, len(df)))

    results = {"source": "wikidata: Iranian humans carrying both fa and en labels"}
    section_transliteration(df, results)
    section_retrieval(df, results)
    section_blocking_distribution(df, results)
    section_homophones(df, results)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nWrote %s" % OUT_PATH)


if __name__ == "__main__":
    main()
