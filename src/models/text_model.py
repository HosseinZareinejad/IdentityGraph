"""
Phase 3 - semantic text comparison features.

Uses the bio/posts vectors already computed and stored in Qdrant by
etl/load_qdrant.py, so no re-embedding is needed at scoring time.

IMPORTANT - placeholder handling. The ETL embeds a literal placeholder
("بدون بیوگرافی" / "بدون پست") for accounts with no bio or no posts, so every
empty-bio account shares an identical vector. Comparing two of them yields a
cosine of 1.0, which looks like overwhelming evidence of a match while
actually meaning "we know nothing about either". This module therefore checks
whether the underlying text was really present and returns None instead. Most
Telegram accounts and many LinkedIn accounts have no posts, so this affects a
large fraction of pairs - it would have been a silent, systematic false-match
generator.

Note on model choice: this still uses the collection's existing
paraphrase-multilingual-MiniLM-L12-v2 vectors. The roadmap's plan to swap in a
stronger multilingual model (multilingual-e5 / BGE-M3) is deliberately
deferred until the ablation in this phase shows what the text path is
actually worth - Phase 2 measured its retrieval recall at ~20% vs ~69% for
names, so upgrading it is unlikely to change the system's shape and the
download is heavy. Decide with the ablation numbers, not by assumption.
"""
import numpy as np


def _cosine(u, v) -> float:
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0.0:
        return 0.0
    return float(np.dot(u, v) / denom)


def bio_similarity(a: dict, b: dict, vectors: dict) -> float | None:
    """Cosine between bio vectors, or None if either bio was actually empty."""
    if not _has_text(a.get("bio")) or not _has_text(b.get("bio")):
        return None
    va, vb = vectors.get(a["record_id"]), vectors.get(b["record_id"])
    if va is None or vb is None:
        return None
    return _cosine(va["bio"], vb["bio"])


def posts_similarity(a: dict, b: dict, vectors: dict) -> float | None:
    """Cosine between pooled-post vectors, or None if either account has no
    posts at all."""
    if not a.get("num_posts") or not b.get("num_posts"):
        return None
    va, vb = vectors.get(a["record_id"]), vectors.get(b["record_id"])
    if va is None or vb is None:
        return None
    return _cosine(va["posts"], vb["posts"])


def _has_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def extract(a: dict, b: dict, vectors: dict) -> dict:
    return {
        "bio_cosine": bio_similarity(a, b, vectors),
        "posts_cosine": posts_similarity(a, b, vectors),
    }


def load_vectors_from_qdrant(client, collection_name: str) -> dict:
    """Pull every point's named vectors into memory once (14k x 384 x 2 floats
    is ~44MB) - far faster than a per-pair round trip during training."""
    vectors = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_vectors=True,
            with_payload=False,
            offset=offset,
        )
        for p in points:
            vectors[str(p.id)] = {
                "bio": np.asarray(p.vector["bio"], dtype=np.float32),
                "posts": np.asarray(p.vector["posts"], dtype=np.float32),
            }
        if offset is None:
            break
    return vectors
