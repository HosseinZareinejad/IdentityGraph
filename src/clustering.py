"""
Phase 4 - turning scored pairs into identity clusters.

The existing resolution.py takes every pair above a threshold, adds it as an
edge, and calls nx.connected_components. That is transitive closure, and it
has a well-known failure mode in entity resolution: ONE false edge welds two
otherwise-correct clusters together, and the damage compounds - a handful of
bad edges can chain hundreds of accounts into a single "black hole" cluster.
Pairwise F1 hides this, because a giant cluster still gets many pairs right.

Three stages here, each measurable separately so the contribution of each is
visible rather than asserted:

  1. threshold          - the naive baseline (what resolution.py does today)
  2. constrained match  - exploit the structural fact that one person holds at
                          most one account per platform, so the correct answer
                          between any two platforms is a 1-to-1 matching, not
                          "every pair over the line". Solved exactly per
                          connected component with the Hungarian algorithm.
  3. cluster hygiene    - a cluster containing two accounts from the same
                          platform, or contradictory birth years, is
                          internally inconsistent regardless of how good each
                          individual edge looked; split it at its weakest link.
"""
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

# Above this size, exact Hungarian assignment on a dense submatrix gets
# expensive (O(n^3)); components this large are rare and greedy matching is a
# 1/2-approximation with far better scaling.
MAX_EXACT_COMPONENT = 150


def threshold_edges(pairs, scores, threshold: float):
    """Baseline: keep every pair scoring above the threshold."""
    return [(a, b, s) for (a, b), s in zip(pairs, scores) if s >= threshold]


def constrained_match(pairs, scores, universe, threshold: float):
    """One-to-one matching per platform pair.

    Each person holds at most one account per platform in this world, so for
    any two platforms the truth is a partial 1-1 matching. Maximising total
    score under that constraint removes the "one popular account matches
    fifteen different people" failure that plain thresholding allows.
    """
    by_platform_pair = defaultdict(list)
    for (a, b), s in zip(pairs, scores):
        if s < threshold:
            continue
        plat_a = universe.accounts[a]["platform"]
        plat_b = universe.accounts[b]["platform"]
        key = tuple(sorted((plat_a, plat_b)))
        by_platform_pair[key].append((a, b, s))

    accepted = []
    for (plat_a, plat_b), edges in by_platform_pair.items():
        accepted.extend(_match_one_platform_pair(edges, universe, plat_a))
    return accepted


def _match_one_platform_pair(edges, universe, platform_a):
    """Max-weight bipartite matching, decomposed by connected component.

    The candidate graph is sparse and breaks into many small components, so
    running exact Hungarian inside each component is both optimal and cheap -
    far better than one dense n x n assignment over all accounts.
    """
    graph = nx.Graph()
    for a, b, s in edges:
        graph.add_edge(a, b, weight=s)

    accepted = []
    for component in nx.connected_components(graph):
        sub = graph.subgraph(component)
        nodes = sorted(component)
        # The graph is bipartite by construction (every edge joins two
        # different platforms), so the sides are just the two platform groups.
        side_a = [n for n in nodes if universe.accounts[n]["platform"] == platform_a]
        side_b = [n for n in nodes if universe.accounts[n]["platform"] != platform_a]
        if not side_a or not side_b:
            continue

        if len(nodes) > MAX_EXACT_COMPONENT:
            accepted.extend(_greedy_match(sub))
            continue

        idx_a = {n: i for i, n in enumerate(side_a)}
        idx_b = {n: i for i, n in enumerate(side_b)}
        cost = np.zeros((len(side_a), len(side_b)), dtype=np.float64)
        for u, v, data in sub.edges(data=True):
            if u in idx_a and v in idx_b:
                cost[idx_a[u], idx_b[v]] = -data["weight"]
            elif v in idx_a and u in idx_b:
                cost[idx_a[v], idx_b[u]] = -data["weight"]

        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            weight = -cost[r, c]
            if weight > 0:  # skip the zero-fill cells that aren't real edges
                accepted.append((side_a[r], side_b[c], weight))
    return accepted


def _greedy_match(sub):
    """1/2-approximation fallback for oversized components."""
    edges = sorted(sub.edges(data=True), key=lambda e: -e[2]["weight"])
    used = set()
    accepted = []
    for u, v, data in edges:
        if u in used or v in used:
            continue
        used.add(u)
        used.add(v)
        accepted.append((u, v, data["weight"]))
    return accepted


def build_clusters(accepted_edges, all_record_ids):
    """Connected components over accepted edges, including singletons."""
    graph = nx.Graph()
    graph.add_nodes_from(all_record_ids)
    for a, b, s in accepted_edges:
        graph.add_edge(a, b, weight=s)
    return [set(c) for c in nx.connected_components(graph)], graph


def enforce_cluster_hygiene(clusters, graph, universe):
    """Split clusters that contradict themselves.

    Even with 1-1 matching per platform pair, transitivity can still produce
    an inconsistent cluster: T1-I1, T2-G1 and I1-G1 are each a valid 1-1 match
    in their own platform pair, yet together they place two Twitter accounts
    in one identity. Rather than trusting the closure, drop the weakest edge
    and re-check until every cluster is internally consistent.
    """
    # Work on a mutable copy: edge removals must persist. Re-deriving each
    # sub-cluster from the ORIGINAL graph would resurrect the edge just
    # removed, and if its endpoints remain connected by another path the same
    # cluster would be split and rebuilt forever.
    working = graph.copy()

    clean = []
    queue = list(clusters)
    while queue:
        cluster = queue.pop()
        if len(cluster) <= 1 or _is_consistent(cluster, universe):
            clean.append(cluster)
            continue

        sub = working.subgraph(cluster)
        if sub.number_of_edges() == 0:
            clean.extend({n} for n in cluster)
            continue

        weakest = min(sub.edges(data=True), key=lambda e: e[2].get("weight", 0.0))
        working.remove_edge(weakest[0], weakest[1])
        for component in nx.connected_components(working.subgraph(cluster)):
            queue.append(set(component))
    return clean


def _is_consistent(cluster, universe) -> bool:
    platforms = []
    birth_years = []
    for rid in cluster:
        acct = universe.accounts[rid]
        platforms.append(acct["platform"])
        by = acct.get("birth_year")
        if by is not None and not (isinstance(by, float) and np.isnan(by)):
            birth_years.append(int(by))

    if len(platforms) != len(set(platforms)):
        return False  # two accounts from the same platform
    if birth_years and (max(birth_years) - min(birth_years)) > 5:
        return False  # irreconcilable ages
    return True


def clusters_to_pairs(clusters) -> set[tuple[str, str]]:
    """Every within-cluster pair, for pairwise scoring against ground truth."""
    pairs = set()
    for cluster in clusters:
        members = sorted(cluster)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs
