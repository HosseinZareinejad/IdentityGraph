"""
Phase 2 - ETL step 3: embed + load into Qdrant.

Builds a single collection ("identity_accounts") holding one point per
platform account across all 4 platforms, with:

  - Two named vectors, "bio" and "posts", both from the SAME global
    sentence-transformer model. Unlike the per-platform graph structure,
    text embeddings ARE directly comparable across platforms (the model
    doesn't know or care which platform a string came from), so this is
    the right place for cross-platform semantic candidate retrieval.
  - A flat, scalar-only payload (platform, username, phonetic key,
    transliterated name, phone/email canonical forms, structured metadata,
    and the graph structural features from etl/graph_features.py) with
    keyword indexes for exact-match blocking. The full nested `posts` list
    is NOT duplicated into Qdrant payload (Qdrant payload should stay flat);
    Phase 3 models that need per-post detail (e.g. a stylometry model) can
    read data/cleaned/{platform}.parquet directly.

One collection (not 4 separate per-platform collections) is a deliberate
choice: cross-platform candidate retrieval needs to search "does any
account on ANY OTHER platform have a similar bio vector", which is a single
query against one collection filtered by platform, not four separate
queries against isolated collections.

`entity_id` is stored in the payload for evaluation/debugging convenience
only (Phase 6). It is the ground-truth label the matching pipeline is
supposed to recover - future retrieval/scoring code must not read it as a
matching feature.
"""
import math
import os
import sys

os.environ["NO_PROXY"] = "localhost,127.0.0.1"  # avoid Windows proxy interference, matches ingestion.py

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import settings  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
GRAPH_FEATURES_DIR = os.path.join(BASE_DIR, "data", "graph_features")
PLATFORMS = ["twitter", "instagram", "telegram", "linkedin"]

COLLECTION_NAME = "identity_accounts"
EMBED_DIM = 384  # matches settings.embedding_model (paraphrase-multilingual-MiniLM-L12-v2)
BATCH_SIZE = 128

PAYLOAD_COLUMNS = [
    "entity_id", "platform", "display_name", "display_name_translit", "username",
    "name_phonetic_key", "bio", "city", "birth_year", "job_title", "education",
    "phone_number", "phone_canonical", "email", "email_normalized", "is_noisy",
    "num_posts", "graph_degree", "clustering_coeff", "avg_neighbor_degree",
    "detected_community_id",
]


def _clean_payload_value(v):
    """Qdrant payload must be JSON-serializable; numpy/pandas scalars aren't."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):  # numpy scalar (int64, float64, bool_, ...)
        v = v.item()
    return v


def build_collection(client: QdrantClient):
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' exists, recreating...")
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "bio": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            "posts": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        },
    )
    for field in ["platform", "username", "phone_canonical", "email_normalized",
                  "city", "name_phonetic_key", "entity_id"]:
        client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=PayloadSchemaType.KEYWORD)


def load_merged(platform: str) -> pd.DataFrame:
    cleaned = pd.read_parquet(os.path.join(CLEANED_DIR, f"{platform}.parquet"))
    graph_feats = pd.read_parquet(os.path.join(GRAPH_FEATURES_DIR, f"{platform}.parquet"))
    return cleaned.merge(graph_feats, on="record_id", how="left")


def embed_and_upsert(client: QdrantClient, model: SentenceTransformer, platform: str) -> int:
    df = load_merged(platform)
    total = len(df)

    for start in range(0, total, BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]

        bio_texts = [t if isinstance(t, str) and t.strip() else "بدون بیوگرافی" for t in batch["bio"]]
        posts_texts = [t if isinstance(t, str) and t.strip() else "بدون پست" for t in batch["posts_text_joined"]]

        bio_vecs = model.encode(bio_texts, show_progress_bar=False)
        posts_vecs = model.encode(posts_texts, show_progress_bar=False)

        points = []
        for i, (_, row) in enumerate(batch.iterrows()):
            payload = {col: _clean_payload_value(row.get(col)) for col in PAYLOAD_COLUMNS}
            points.append(PointStruct(
                id=row["record_id"],
                vector={"bio": bio_vecs[i].tolist(), "posts": posts_vecs[i].tolist()},
                payload=payload,
            ))
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  {platform}: upserted {min(start + BATCH_SIZE, total)}/{total}")

    return total


def run():
    print(f"Connecting to Qdrant at {settings.qdrant_url}...")
    client = QdrantClient(settings.qdrant_url, timeout=60.0)
    client.get_collections()  # fail fast if unreachable

    print(f"Loading embedding model: {settings.embedding_model}...")
    model = SentenceTransformer(settings.embedding_model)

    print(f"(Re)creating collection '{COLLECTION_NAME}'...")
    build_collection(client)

    print("Embedding and upserting platform accounts...")
    total = 0
    for platform in PLATFORMS:
        total += embed_and_upsert(client, model, platform)

    info = client.get_collection(COLLECTION_NAME)
    print(f"\nDone. {total} points loaded. Collection point count: {info.points_count}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run()
