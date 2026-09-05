"""
Phase 2 - ETL validation suite.

Codifies every correctness/quality check for the Phase 1 -> Phase 2 pipeline
so "the ETL ran" and "the ETL is correct" are two different, testable claims.

Checks:
  1. Referential integrity  - cleaned tables, graph features, ground-truth
     pairs and Qdrant points all agree on the same record_id universe.
  2. Field coverage         - actual per-platform null rates vs. the coverage
     matrix the generator was configured with (catches silent ETL data loss).
  3. Graph structure        - Louvain communities vs. the ground-truth
     communities that drove edge generation (Adjusted Rand Index). Confirms
     Phase 1 produced real community structure, not noise.
  4. Qdrant schema          - named vectors + payload indexes actually exist.
  5. Candidate recall       - THE quality metric: for known true
     cross-platform pairs, does semantic search actually surface the
     counterpart? This is what caught the template-homogeneity bug in the
     first version of the generator, where recall@10 was 5%.

Exit code is non-zero if any hard check (1, 4) fails, so this can gate later
phases. Soft metrics (2, 3, 5) are reported, not enforced.
"""
import os
import sys

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rapidfuzz.distance import JaroWinkler
from sklearn.metrics import adjusted_rand_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.append(BASE_DIR)
sys.path.append(DATA_DIR)

from src.config import settings  # noqa: E402
import generate_synthetic_world as gen  # noqa: E402  (coverage matrix source of truth)

CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
GRAPH_FEATURES_DIR = os.path.join(DATA_DIR, "graph_features")
PLATFORM_DUMP_DIR = os.path.join(DATA_DIR, "platform_dumps")
COLLECTION_NAME = "identity_accounts"
PLATFORMS = ["twitter", "instagram", "telegram", "linkedin"]

RECALL_SAMPLE_SIZE = 300


def check_referential_integrity() -> list[str]:
    print("\n[1] Referential integrity")
    failures = []
    all_cleaned_ids = set()

    for platform in PLATFORMS:
        raw = pd.read_parquet(os.path.join(PLATFORM_DUMP_DIR, f"{platform}.parquet"))
        cleaned = pd.read_parquet(os.path.join(CLEANED_DIR, f"{platform}.parquet"))
        feats = pd.read_parquet(os.path.join(GRAPH_FEATURES_DIR, f"{platform}.parquet"))

        raw_ids, cleaned_ids, feat_ids = set(raw["record_id"]), set(cleaned["record_id"]), set(feats["record_id"])
        all_cleaned_ids |= cleaned_ids

        if raw_ids != cleaned_ids:
            failures.append(f"{platform}: cleaned ids != raw ids ({len(raw_ids)} vs {len(cleaned_ids)})")
        if not cleaned_ids.issubset(feat_ids):
            failures.append(f"{platform}: {len(cleaned_ids - feat_ids)} accounts missing graph features")
        print(f"  {platform}: raw={len(raw_ids)} cleaned={len(cleaned_ids)} graph_features={len(feat_ids)} OK")

    gt = pd.read_parquet(os.path.join(DATA_DIR, "ground_truth_pairs.parquet"))
    dangling = (~gt["record_id_a"].isin(all_cleaned_ids)).sum() + (~gt["record_id_b"].isin(all_cleaned_ids)).sum()
    if dangling:
        failures.append(f"ground_truth_pairs: {dangling} dangling record_id references")
    print(f"  ground_truth_pairs: {len(gt)} pairs, {dangling} dangling refs")

    hn = pd.read_parquet(os.path.join(DATA_DIR, "hard_negative_pairs.parquet"))
    print(f"  hard_negative_pairs: {len(hn)} twin pairs")
    return failures


def check_field_coverage():
    print("\n[2] Field coverage vs. generator configuration")
    expected = {
        "city": gen.CITY_COVERAGE,
        "birth_year": gen.BIRTH_YEAR_COVERAGE,
        "job_title": gen.JOB_EDU_COVERAGE,
        "phone_number": gen.PHONE_COVERAGE,
        "email": gen.EMAIL_COVERAGE,
    }
    for platform in PLATFORMS:
        df = pd.read_parquet(os.path.join(CLEANED_DIR, f"{platform}.parquet"))
        parts = []
        for field, table in expected.items():
            actual = df[field].notna().mean()
            exp = table[platform]
            flag = "" if abs(actual - exp) < 0.05 else "  <-- DRIFT"
            parts.append(f"{field}={actual:.2f}(exp {exp:.2f}){flag}")
        print(f"  {platform}: " + ", ".join(parts))


def check_graph_structure():
    print("\n[3] Graph structure (Louvain vs. ground-truth communities)")
    registry = pd.read_parquet(os.path.join(DATA_DIR, "real_identity_registry.parquet"))
    for platform in PLATFORMS:
        feats = pd.read_parquet(os.path.join(GRAPH_FEATURES_DIR, f"{platform}.parquet"))
        cleaned = pd.read_parquet(os.path.join(CLEANED_DIR, f"{platform}.parquet"))
        merged = feats.merge(cleaned[["record_id", "entity_id"]], on="record_id")
        merged = merged.merge(registry[["entity_id", "community_id"]], on="entity_id")
        ari = adjusted_rand_score(merged["community_id"], merged["detected_community_id"])
        print(f"  {platform}: ARI={ari:.3f}, mean degree={feats['graph_degree'].mean():.1f}, "
              f"mean clustering={feats['clustering_coeff'].mean():.3f}")


def check_qdrant_schema(client) -> list[str]:
    print("\n[4] Qdrant schema")
    failures = []
    try:
        info = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        return [f"collection '{COLLECTION_NAME}' unreachable: {e}"]

    vectors = info.config.params.vectors
    for name in ("bio", "posts"):
        if name not in vectors:
            failures.append(f"missing named vector '{name}'")
    print(f"  points={info.points_count}, vectors={list(vectors.keys())}")

    expected_indexes = {"platform", "username", "phone_canonical", "email_normalized",
                        "city", "name_phonetic_key", "entity_id"}
    actual_indexes = set((info.payload_schema or {}).keys())
    missing = expected_indexes - actual_indexes
    if missing:
        failures.append(f"missing payload indexes: {sorted(missing)}")
    print(f"  payload indexes present: {sorted(actual_indexes)}")

    total_cleaned = sum(len(pd.read_parquet(os.path.join(CLEANED_DIR, f"{p}.parquet"))) for p in PLATFORMS)
    if info.points_count != total_cleaned:
        failures.append(f"point count {info.points_count} != cleaned rows {total_cleaned}")
    return failures


def check_retrieval_recall(client):
    """THE quality metric for Phase 2.

    For known true cross-platform pairs, measure whether each retrieval path
    the ETL enables would surface the counterpart, and what the UNION of all
    paths recovers. The union is the hard ceiling on the whole system: a pair
    that no path retrieves can never be scored, no matter how good the Phase-3
    models are.

    Thresholds here are illustrative (Phase 4 tunes blocking properly); the
    point is the relative contribution of each path.
    """
    print(f"\n[5] Cross-platform retrieval recall by path (sample={RECALL_SAMPLE_SIZE})")
    gt = pd.read_parquet(os.path.join(DATA_DIR, "ground_truth_pairs.parquet"))
    sample = gt.sample(min(RECALL_SAMPLE_SIZE, len(gt)), random_state=1)

    cleaned = {p: pd.read_parquet(os.path.join(CLEANED_DIR, f"{p}.parquet")).set_index("record_id")
               for p in PLATFORMS}

    hits_by_path = {k: 0 for k in
                    ["phonetic_key_exact", "name_fuzzy_085", "username_fuzzy_080",
                     "phone_or_email_exact", "bio_vector_top50", "posts_vector_top50", "UNION"]}
    evaluated = 0

    for _, row in sample.iterrows():
        src_id, tgt_id = row["record_id_a"], row["record_id_b"]
        src_platform, tgt_platform = row["platform_a"], row["platform_b"]
        a = cleaned[src_platform].loc[src_id]
        b = cleaned[tgt_platform].loc[tgt_id]

        found = {}
        found["phonetic_key_exact"] = bool(a["name_phonetic_key"]) and a["name_phonetic_key"] == b["name_phonetic_key"]
        found["name_fuzzy_085"] = JaroWinkler.normalized_similarity(
            str(a["display_name"] or ""), str(b["display_name"] or "")) >= 0.85
        found["username_fuzzy_080"] = JaroWinkler.normalized_similarity(
            str(a["username"] or ""), str(b["username"] or "")) >= 0.80
        found["phone_or_email_exact"] = bool(
            (a["phone_canonical"] and a["phone_canonical"] == b["phone_canonical"])
            or (a["email_normalized"] and a["email_normalized"] == b["email_normalized"])
        )

        pts = client.retrieve(COLLECTION_NAME, ids=[src_id], with_vectors=True)
        if not pts:
            continue
        for vector_name, key in (("bio", "bio_vector_top50"), ("posts", "posts_vector_top50")):
            vec_hits = client.query_points(
                collection_name=COLLECTION_NAME,
                query=pts[0].vector[vector_name],
                using=vector_name,
                limit=50,
                with_payload=False,
                query_filter=Filter(must=[FieldCondition(key="platform", match=MatchValue(value=tgt_platform))]),
            ).points
            found[key] = tgt_id in [str(h.id) for h in vec_hits]

        evaluated += 1
        for key, hit in found.items():
            if hit:
                hits_by_path[key] += 1
        if any(found.values()):
            hits_by_path["UNION"] += 1

    for key in hits_by_path:
        label = "UNION (system ceiling)" if key == "UNION" else key
        print(f"  {label:24s} {hits_by_path[key] / evaluated:6.1%}  ({hits_by_path[key]}/{evaluated})")
    return hits_by_path


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 70)
    print("Phase 2 ETL validation")
    print("=" * 70)

    failures = []
    failures += check_referential_integrity()
    check_field_coverage()
    check_graph_structure()

    client = QdrantClient(settings.qdrant_url, timeout=60.0)
    failures += check_qdrant_schema(client)
    check_retrieval_recall(client)

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED ({len(failures)} hard check(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All hard checks passed.")


if __name__ == "__main__":
    main()
