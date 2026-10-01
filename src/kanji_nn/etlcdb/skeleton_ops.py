import numpy as np
from collections import defaultdict, deque


def degrees(glyph):
    return glyph["skeleton:degrees"]


def paths(glyph):
    return glyph["skeleton:paths"]


def lengths(glyph):
    return glyph["skeleton:lengths"]


def edt(glyph):
    """Per-pixel EDT."""
    edt = glyph["edt"] # matrix in x/y layout
    xy = glyph["skeleton:xy"]
    x, y = xy[:, 0], xy[:, 1]
    return edt[x, y]


def edt_radius(glyph, degree=2):
     return 2 * np.median(edt(glyph)[degrees(glyph) == degree])


def pixel_paths(glyph):
    """Return reverse lookup: {pixel index: [path index]}."""
    paths_ = glyph["skeleton:paths"]
    pixel_branches = defaultdict(list)
    for branch_idx, pixel_indices in enumerate(paths_):
        for pixel_idx in pixel_indices:
            pixel_branches[int(pixel_idx)].append(branch_idx)

    return pixel_branches


def junction_nodes(glyph):
    return set(np.where(degrees(glyph) > 2)[0])


def endpoint_nodes(glyph):
    return set(np.where(degrees(glyph) == 1)[0])


def directed_endpoints(glyph):
    endpoints = lambda path: (int(path[0]), int(path[-1]))
    return [endpoints(path) for path in paths(glyph)]


def undirected_endpoints(glyph):
    return [tuple(sorted((pair))) for pair in directed_endpoints(glyph)]


def distance(glyph, a, b):
    "Return Euclidean distance between two nodes."
    xy = glyph["skeleton:xy"][(a, b), :]
    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return float(distances[0])


def parallel_paths(glyph):
    """
    """
    indices = {}
    for path_index, endpoints_ in enumerate(undirected_endpoints(glyph)):
        xs = indices.setdefault(endpoints_, [])
        xs.append(path_index)

    return [
        [endpoints, path_indices]
        for endpoints, path_indices in indices.items()
        if len(path_indices) > 1
    ]


def junction_graph(glyph, radius=None):
    """
    """
    # Set of all junctions (maps indices to regular integers):
    # junctions :: {int}
    junctions = junction_nodes(glyph)

    # Deriving radius purely on the fly from skeleton path pixels
    # (Using degrees == 2 correctly samples path interiors)
    if radius is None:
        radius = edt_radius(glyph, degree=2)

    # Map one pixel to its neighboring junctions (as set).
    # adjacency :: {int: {int}}
    adjacency = defaultdict(set)
    endpoints = zip(undirected_endpoints(glyph), lengths(glyph))

    for (a, b), length in endpoints:
        if (
            a != b
            and a in junctions
            and b in junctions
            and length <= radius
        ):
            adjacency[a].add(b)
            adjacency[b].add(a)

    return adjacency


def connected_components(glyph, adjacency):
    """
    Find connected components of an undirected graph using breadth-first search.

    Parameters
    ----------
    adjacency : mapping
        Mapping from each node to a set of neighboring nodes.

    Returns
    -------
    list of list
        Connected components, with each component represented as a list of
        nodes.
    """

    junctions = junction_nodes(glyph)
    visited = set()
    components = []

    for current in junctions:
        if current in visited:
            continue

        component = []
        queue = deque([current])
        visited.add(current)

        while queue:
            node = queue.popleft()
            component.append(node)

            for neighbour in adjacency[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        components.append(component)

    return components


def junction_clusters(glyph, radius=None):
    """
    Group nearby skeleton junctions into connected clusters.

    A junction is a skeleton node with degree greater than 2. Two junctions
    are considered directly connected when they are the endpoints of the same
    skeleton branch and that branch has length less than or equal to
    ``radius``. Junctions are then grouped by finding the connected components
    of this junction-to-junction graph.

    This allows a visually or geometrically single junction region that has
    been represented by multiple nearby junction nodes to be treated as one
    cluster. Clustering is based on skeleton path connectivity and path
    length, rather than direct Euclidean distance between junction pixels.

    Parameters
    ----------
    radius : float, optional
        Maximum skeleton branch length for connecting two junctions. If
        ``None``, the radius is estimated as twice the median Euclidean
        distance-transform (EDT) value of degree-2 skeleton pixels. Degree-2
        pixels represent path interiors and are used to obtain a typical
        local scale for the skeleton while avoiding junction and endpoint
        pixels.

    Returns
    -------
    list of list of int
        Connected junction clusters. Each inner list contains the pixel/node
        indices of junctions belonging to the same cluster. A junction with
        no qualifying short connection to another junction forms a
        single-element cluster.

    Notes
    -----
    The method uses skeleton path lengths when determining whether junctions
    are connected. Consequently, the threshold is based on distance along the
    skeleton rather than straight-line Euclidean distance between junctions.

    The automatically estimated radius is intended to provide a scale based
    on the typical thickness of the underlying object, as represented by the
    EDT values along ordinary (degree-2) skeleton paths.
    """

    adjacency = junction_graph(glyph, radius)
    clusters = connected_components(glyph, adjacency)
    return [sorted(map(int, cluster)) for cluster in clusters]


def cluster_paths(glyph, clusters):
    """
    Associate external paths with each junction cluster.

    Paths whose two endpoints belong to the same cluster are considered
    internal and are excluded.
    """
    directed = directed_endpoints(glyph)

    results = []

    for cluster in clusters:
        junctions = set(cluster)
        incoming = []
        outgoing = []

        for path_index, (start, end) in enumerate(directed):
            start_in = start in junctions
            end_in = end in junctions

            if end_in and not start_in:
                incoming.append(path_index)

            if start_in and not end_in:
                outgoing.append(path_index)

        results.append({
            "junctions": sorted(map(int, junctions)),
            "incoming": incoming,
            "outgoing": outgoing,
        })

    return results
