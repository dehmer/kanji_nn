from collections import Counter
import numpy as np
from skan.csr import Skeleton
import networkx as nx
from .connected_features import fold


def junction_branches(G, junction):
    """Return the edges incident to ``junction``, oriented away from it.

    Each branch's ``coordinates`` starts at the junction coordinate and ends
    at its other graph node.  Keeping that orientation makes comparisons at a
    junction independent of the endpoint order NetworkX returns for an edge.
    """
    branches = []
    for u, v, key, data in G.edges(junction, keys=True, data=True):
        if u == v:
            # A self-loop does not define two distinct branches at a junction.
            continue

        if u == junction:
            other, coordinates = v, data["coordinates"]
        else:
            other, coordinates = u, data["coordinates"][::-1]

        if len(coordinates) < 2:
            continue

        branches.append({
            "edge": (u, v, key),
            "node": other,
            "coordinates": coordinates,
        })

    return branches


def are_collinear_at_junction(branch_a, branch_b, *, atol=1e-8):
    """Whether the first raster segment of two outward branches shares a line."""
    a = branch_a["coordinates"]
    b = branch_b["coordinates"]
    # Both polylines begin at the same junction.  The next pixel supplies the
    # direction of each branch immediately adjacent to it.
    direction_a = a[1] - a[0]
    direction_b = b[1] - b[0]
    cross_product = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    return bool(np.isclose(cross_product, 0.0, atol=atol))


def connected_branch_pairs(G, junctions=None):
    """Return every pair of branches meeting at a junction.

    A result has the form ``{"junction", "a", "b", "collinear"}``.
    Branches are oriented outward from ``junction``; consequently the tested
    points are ``a[1]``, ``junction``, and ``b[1]``.  This is equivalent to
    orienting A into the junction and B out of it.
    """
    if junctions is None:
        junctions = [node for node, _ in query_junctions(G)]

    pairs = []
    for junction in junctions:
        branches = junction_branches(G, junction)
        for a_index, branch_a in enumerate(branches):
            for branch_b in branches[a_index + 1:]:
                pairs.append({
                    "junction": junction,
                    "a": branch_a,
                    "b": branch_b,
                    "collinear": are_collinear_at_junction(branch_a, branch_b),
                })

    return pairs

def query_edges(G, predicate):
    """Returns a generator of edges matching the predicate function."""
    # predicate expects (u, v, key, data)
    return filter(lambda e: predicate(*e), G.edges(keys=True, data=True))

def query_nodes(G, predicate):
    """Returns a generator of nodes matching the predicate function."""
    # predicate expects (node_id, data)
    return filter(lambda n: predicate(*n), G.nodes(data=True))

def query_junctions(G):
    return list(query_nodes(G, lambda key, _: G.degree[key] > 2))

def consolidate_graph(glyph):
    pixel_graph = glyph["pixel_graph"]

    parallel_branches = pixel_graph.parallel_branches()
    if len(parallel_branches):
        print(glyph["literal"], glyph["id"], len(parallel_branches))
        for branch in parallel_branches:
            nodes, branches = branch
            print("  ", branch, pixel_graph.distance(*nodes), pixel_graph.coords[nodes, :])

    return glyph
