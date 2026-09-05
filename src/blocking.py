"""
Phase 4 - multi-path candidate generation (blocking).

Blocking sets the hard ceiling on the whole system: a true pair that never
enters the candidate set can never be scored, no matter how good the fusion
model is. Phase 3 used a single path (top-10 by name similarity) and reached
68.1% recall, while Phase 2 measured the union of all available paths at
92.7% - roughly 25 points left on the table purely by not looking.

Each path here targets a different failure mode of the others:

  phonetic     exact match on the homophone-collapsed key - catches ز/ذ,
               س/ص/ث, ت/ط confusions that fuzzy string distance scores low
  name_fuzzy   top-K Jaro-Winkler - the general-purpose path
  username     top-K on handles - works when display names diverge entirely
  username_skel same, with Latin vowels removed from both sides: one person
               romanises their own name differently per site, and the vowels
               are exactly the part Persian script never specified
  cross_field  A's handle vs B's transliterated name and vice versa - the
               Twitter-handle-to-LinkedIn-legal-name case, where neither the
               name path nor the username path fires
  contact      exact phone/email - rare but decisive
  vector       top-K bio/posts embedding similarity - the only path that can
               fire when the name is a pseudonym sharing nothing with the
               real name

Every path records itself on the pair it produced, so run_resolution.py can
report each path's marginal contribution and drop any that doesn't earn its
cost.
"""
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import process as rf_process
from rapidfuzz.distance import JaroWinkler

from src.models.name_model import strip_digits, strip_titles
from src.text_utils import latin_skeleton, transliterate

PLATFORMS = ["twitter", "instagram", "telegram", "linkedin"]

# A blocking key shared by more than this many accounts produces more pairs
# than it is worth (quadratic in block size) and is almost never
# discriminative - standard practice is to drop such blocks.
MAX_BLOCK_SIZE = 60

DEFAULT_CONFIG = {
    "name_fuzzy_top_k": 10,
    "username_top_k": 10,
    "username_skeleton_top_k": 10,
    "cross_field_top_k": 5,
    "vector_top_k": 5,
    "use_phonetic": True,
    "use_contact": True,
    "use_vector": True,
}


def _accounts_by_platform(universe) -> dict[str, list[str]]:
    by_platform = {p: [] for p in PLATFORMS}
    for rid, acct in universe.accounts.items():
        by_platform[acct["platform"]].append(rid)
    return by_platform


def _exact_key_pairs(universe, by_platform, key_fn, path_name, pairs):
    """Generic exact-match blocker: group accounts by a key, emit all
    cross-platform pairs inside each (small enough) group."""
    blocks = defaultdict(list)
    for rid, acct in universe.accounts.items():
        key = key_fn(acct)
        if key:
            blocks[key].append(rid)

    emitted = 0
    for key, members in blocks.items():
        if len(members) > MAX_BLOCK_SIZE:
            continue
        for rid_a, rid_b in itertools.combinations(sorted(members), 2):
            if universe.accounts[rid_a]["platform"] == universe.accounts[rid_b]["platform"]:
                continue  # cross-platform linkage only
            pairs[(rid_a, rid_b)].add(path_name)
            emitted += 1
    return emitted


def _top_k_pairs(ids_a, ids_b, strings_a, strings_b, top_k, path_name, pairs, universe):
    """Top-K nearest by Jaro-Winkler, computed with rapidfuzz's C backend."""
    if not ids_a or not ids_b or top_k <= 0:
        return 0
    sim = rf_process.cdist(strings_a, strings_b, scorer=JaroWinkler.normalized_similarity,
                           workers=-1, dtype=np.float32)
    k = min(top_k, sim.shape[1])
    top = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    emitted = 0
    for row_idx, cols in enumerate(top):
        rid_a = ids_a[row_idx]
        for col in cols:
            rid_b = ids_b[col]
            key = (rid_a, rid_b) if rid_a < rid_b else (rid_b, rid_a)
            pairs[key].add(path_name)
            emitted += 1
    return emitted


def _vector_top_k_pairs(ids_a, ids_b, vectors, vector_name, top_k, path_name, pairs):
    """Top-K by cosine similarity, done as one matmul rather than per-query
    round trips to Qdrant."""
    if not ids_a or not ids_b or top_k <= 0:
        return 0
    mat_a = np.stack([vectors[r][vector_name] for r in ids_a])
    mat_b = np.stack([vectors[r][vector_name] for r in ids_b])
    mat_a /= (np.linalg.norm(mat_a, axis=1, keepdims=True) + 1e-9)
    mat_b /= (np.linalg.norm(mat_b, axis=1, keepdims=True) + 1e-9)
    sim = mat_a @ mat_b.T

    k = min(top_k, sim.shape[1])
    top = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    emitted = 0
    for row_idx, cols in enumerate(top):
        rid_a = ids_a[row_idx]
        for col in cols:
            rid_b = ids_b[col]
            key = (rid_a, rid_b) if rid_a < rid_b else (rid_b, rid_a)
            pairs[key].add(path_name)
            emitted += 1
    return emitted


def generate_candidates(universe, vectors: dict | None = None, config: dict | None = None,
                        verbose: bool = True) -> pd.DataFrame:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    vectors = vectors or {}
    by_platform = _accounts_by_platform(universe)
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)

    if cfg["use_phonetic"]:
        n = _exact_key_pairs(universe, by_platform,
                             lambda a: a.get("name_phonetic_key"), "phonetic", pairs)
        if verbose:
            print(f"  phonetic block:      +{n} pairs")

    if cfg["use_contact"]:
        n_phone = _exact_key_pairs(universe, by_platform,
                                   lambda a: a.get("phone_canonical"), "contact", pairs)
        n_email = _exact_key_pairs(universe, by_platform,
                                   lambda a: a.get("email_normalized"), "contact", pairs)
        if verbose:
            print(f"  contact block:       +{n_phone + n_email} pairs")

    for plat_a, plat_b in itertools.combinations(PLATFORMS, 2):
        ids_a, ids_b = by_platform[plat_a], by_platform[plat_b]
        acc = universe.accounts

        names_a = [strip_titles(str(acc[r]["display_name"] or "")) for r in ids_a]
        names_b = [strip_titles(str(acc[r]["display_name"] or "")) for r in ids_b]
        _top_k_pairs(ids_a, ids_b, names_a, names_b, cfg["name_fuzzy_top_k"], "name_fuzzy", pairs, universe)

        users_a = [strip_digits(str(acc[r]["username"] or "")) for r in ids_a]
        users_b = [strip_digits(str(acc[r]["username"] or "")) for r in ids_b]
        _top_k_pairs(ids_a, ids_b, users_a, users_b, cfg["username_top_k"], "username", pairs, universe)

        # Same handles compared without their vowels. One person romanises
        # their own name differently on different sites ("hoseyni" here,
        # "hosseini" there) because Persian script never wrote the vowels
        # down; the skeleton is what the two spellings genuinely share. Kept
        # as its own path rather than replacing the literal one so
        # path_contributions() can show what it actually adds.
        skel_users_a = [latin_skeleton(u) for u in users_a]
        skel_users_b = [latin_skeleton(u) for u in users_b]
        _top_k_pairs(ids_a, ids_b, skel_users_a, skel_users_b, cfg["username_skeleton_top_k"],
                     "username_skel", pairs, universe)

        # cross-field, both directions: handle on one side vs transliterated
        # legal name on the other. transliterate() emits no vowels, so this
        # comparison is only meaningful in skeleton space.
        translit_a = [latin_skeleton(transliterate(n)) for n in names_a]
        translit_b = [latin_skeleton(transliterate(n)) for n in names_b]
        _top_k_pairs(ids_a, ids_b, skel_users_a, translit_b, cfg["cross_field_top_k"],
                     "cross_field", pairs, universe)
        _top_k_pairs(ids_b, ids_a, skel_users_b, translit_a, cfg["cross_field_top_k"],
                     "cross_field", pairs, universe)

        if cfg["use_vector"] and vectors:
            _vector_top_k_pairs(ids_a, ids_b, vectors, "bio", cfg["vector_top_k"], "vector_bio", pairs)
            _vector_top_k_pairs(ids_a, ids_b, vectors, "posts", cfg["vector_top_k"], "vector_posts", pairs)

    rows = []
    for (rid_a, rid_b), paths in pairs.items():
        a, b = universe.accounts[rid_a], universe.accounts[rid_b]
        if a["platform"] == b["platform"]:
            continue
        rows.append({
            "record_id_a": rid_a, "record_id_b": rid_b,
            "platform_a": a["platform"], "platform_b": b["platform"],
            "entity_id_a": a["entity_id"], "entity_id_b": b["entity_id"],
            "label": int(a["entity_id"] == b["entity_id"]),
            "paths": ",".join(sorted(paths)),
        })
    return pd.DataFrame(rows)


def path_contributions(candidates: pd.DataFrame, n_true_pairs: int) -> pd.DataFrame:
    """Per-path recall, plus the marginal recall each path adds that no other
    path found - the number that decides whether a path earns its cost."""
    all_paths = sorted({p for paths in candidates["paths"] for p in paths.split(",")})
    positives = candidates[candidates["label"] == 1]

    rows = []
    for path in all_paths:
        has_path = candidates["paths"].str.contains(rf"(?:^|,){path}(?:,|$)", regex=True)
        pos_has_path = positives["paths"].str.contains(rf"(?:^|,){path}(?:,|$)", regex=True)
        only_this = positives["paths"] == path
        rows.append({
            "path": path,
            "candidates": int(has_path.sum()),
            "true_pairs_found": int(pos_has_path.sum()),
            "recall": pos_has_path.sum() / n_true_pairs,
            "unique_true_pairs": int(only_this.sum()),
        })
    return pd.DataFrame(rows).sort_values("recall", ascending=False)
