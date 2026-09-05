"""
Phase 2 - ETL step 2: per-platform graph structural features.

Loads each platform's edge list (data/social_graphs/{platform}_edges.parquet,
built in Phase 1) and computes, per account:
  - graph_degree
  - clustering_coeff
  - avg_neighbor_degree
  - detected_community_id (Louvain modularity communities)

These are PLATFORM-AGNOSTIC NUMERIC features - comparable across platforms
even though the graphs themselves (and any node2vec embedding trained on
them) are not. This is deliberate: node2vec embeddings for the Twitter graph
and the LinkedIn graph live in unrelated vector spaces, so a Phase-3 "graph
model" must not cosine-compare raw embeddings across platforms. It CAN use
these structural numbers as comparable inputs to the metadata/fusion model,
or use the graphs directly for label-propagation once seed matches exist.
See the module docstring in data/generate_synthetic_world.py for the full
rationale.

As a sanity check that Phase 1's community-correlated graph generation
actually produced structure (and not just noise), this script also compares
the Louvain-DETECTED communities against the GROUND-TRUTH community_id that
drove edge generation, via Adjusted Rand Index. This is a validation signal
for the data generator, not a feature used by any model.
"""
import os
import sys

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
GRAPH_DIR = os.path.join(DATA_DIR, "social_graphs")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
REGISTRY_PATH = os.path.join(DATA_DIR, "real_identity_registry.parquet")
GRAPH_FEATURES_DIR = os.path.join(DATA_DIR, "graph_features")
PLATFORMS = ["twitter", "instagram", "telegram", "linkedin"]


def load_platform_graph(platform: str, record_ids: list[str]) -> nx.Graph:
    edges = pd.read_parquet(os.path.join(GRAPH_DIR, f"{platform}_edges.parquet"))
    G = nx.Graph()
    G.add_nodes_from(record_ids)  # ensures isolated/degree-0 accounts are still represented
    G.add_edges_from(edges[["record_id_a", "record_id_b"]].itertuples(index=False, name=None))
    return G


def compute_structural_features(G: nx.Graph) -> pd.DataFrame:
    degree = dict(G.degree())
    clustering = nx.clustering(G)
    avg_neighbor_degree = nx.average_neighbor_degree(G)

    communities = nx.community.louvain_communities(G, seed=42)
    detected_community = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            detected_community[node] = comm_id

    rows = [
        {
            "record_id": node,
            "graph_degree": degree.get(node, 0),
            "clustering_coeff": round(clustering.get(node, 0.0), 4),
            "avg_neighbor_degree": round(avg_neighbor_degree.get(node, 0.0), 3),
            "detected_community_id": detected_community.get(node, -1),
        }
        for node in G.nodes()
    ]
    return pd.DataFrame(rows)


def validate_against_ground_truth(features_df, cleaned_df, registry_df) -> float:
    merged = features_df.merge(cleaned_df[["record_id", "entity_id"]], on="record_id")
    merged = merged.merge(registry_df[["entity_id", "community_id"]], on="entity_id")
    return adjusted_rand_score(merged["community_id"], merged["detected_community_id"])


def run() -> dict:
    os.makedirs(GRAPH_FEATURES_DIR, exist_ok=True)
    registry_df = pd.read_parquet(REGISTRY_PATH)
    ari_scores = {}
    for platform in PLATFORMS:
        cleaned_df = pd.read_parquet(os.path.join(CLEANED_DIR, f"{platform}.parquet"))
        record_ids = cleaned_df["record_id"].tolist()
        G = load_platform_graph(platform, record_ids)
        features_df = compute_structural_features(G)
        features_df.to_parquet(os.path.join(GRAPH_FEATURES_DIR, f"{platform}.parquet"), index=False)

        ari = validate_against_ground_truth(features_df, cleaned_df, registry_df)
        ari_scores[platform] = ari
        print(
            f"  {platform}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
            f"{len(nx.community.louvain_communities(G, seed=42))} detected communities, "
            f"ARI vs. ground-truth community = {ari:.3f}"
        )
    return ari_scores


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("Computing per-platform graph structural features...")
    run()
