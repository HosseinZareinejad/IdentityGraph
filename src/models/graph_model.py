"""
Phase 3 - graph structural comparison features.

WHY THERE IS NO node2vec HERE (deviation from the roadmap, deliberate):

The roadmap listed node2vec as the graph model. Phase 1/2 established that
each platform has its OWN graph, and a node2vec embedding trained on the
Twitter graph and one trained on the LinkedIn graph live in unrelated vector
spaces - the axes mean different things, so the cosine between them is
meaningless. Comparing them anyway would produce confident-looking numbers
with no content. Embedding-space alignment needs known anchor pairs, which
are precisely what we are trying to discover, so it is circular at this
stage.

What IS valid across platforms is platform-agnostic structural description:
how connected an account is, how cliquey its neighbourhood is, how its degree
compares to its neighbours'. If a person is a hub in their Twitter circle
they tend to be a hub in their Instagram circle, because both graphs are
generated from the same underlying real-world community structure (ARI ~0.28
against ground truth, measured in etl/validate.py).

These are weak features on their own and are expected to earn little weight
in the ablation. The strong graph signal - propagating confidence across
already-matched neighbours - requires seed matches from the other modalities
first, so it belongs in Phase 4 alongside candidate generation and
constrained clustering, not here.
"""
import numpy as np


def _ratio_similarity(x: float, y: float) -> float | None:
    """Scale-free comparison of two positive quantities: 1.0 when equal,
    decaying as they diverge in ratio."""
    if x is None or y is None:
        return None
    x, y = float(x), float(y)
    if x <= 0 and y <= 0:
        return None
    denom = max(x, y)
    if denom <= 0:
        return None
    return float(min(x, y) / denom)


def degree_similarity(a: dict, b: dict) -> float | None:
    return _ratio_similarity(a.get("graph_degree"), b.get("graph_degree"))


def clustering_similarity(a: dict, b: dict) -> float | None:
    ca, cb = a.get("clustering_coeff"), b.get("clustering_coeff")
    if ca is None or cb is None:
        return None
    return float(np.exp(-abs(float(ca) - float(cb)) / 0.15))


def neighbor_degree_similarity(a: dict, b: dict) -> float | None:
    return _ratio_similarity(a.get("avg_neighbor_degree"), b.get("avg_neighbor_degree"))


def extract(a: dict, b: dict) -> dict:
    return {
        "graph_degree_sim": degree_similarity(a, b),
        "graph_clustering_sim": clustering_similarity(a, b),
        "graph_neighbor_degree_sim": neighbor_degree_similarity(a, b),
    }
