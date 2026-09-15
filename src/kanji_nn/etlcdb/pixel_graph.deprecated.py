from collections import Counter
import numpy as np
from skan.csr import Skeleton
from scipy.spatial.distance import cdist


class PixelGraph:
    def __init__(self, s: Skeleton, edt):
        self.branches = [s.path(i) for i in range(0, s.n_paths)]
        # Flip row/column layout to x/y coordinate layout:
        self.coords = np.column_stack([s.coordinates[:, 1], s.coordinates[:, 0]])
        self.degrees = s.degrees
        self.lengths = s.path_lengths()
        self.edt = edt

        with np.printoptions(precision=2, suppress=True, threshold=np.inf):
            # FIXME: all 1's for 姶 cac3a47c-d567-4bad-9c94-64c1eb60b2cd
            print(edt[np.where(edt != 0.0)])

    def junctions(self):
        return self.degrees > 2

    def endpoints(self):
        return self.degrees == 1

    def distance(self, a, b):
        "Return Euclidean distance between two nodes."
        coords = self.coords[(a, b), :]
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        return float(distances[0])

    def parallel_branches(self):
        """

        """
        indices = {}
        for i, branch in enumerate(self.branches):
            endpoints = tuple(sorted((int(branch[0]), int(branch[-1]))))
            xs = indices.setdefault(endpoints, [])
            xs.append(i)

        return [[k, v] for k, v in indices.items() if len(v) > 1]

type sgraph = Skeleton


def sgraph_from_image(image) -> sgraph:
    """Create Skeleton-based sgraph from binary PIL image."""
    return Skeleton(image)



def distance(g: sgraph, node_a, node_b):
    "Return Euclidean distance between two nodes."
    coords = g.coordinates[(node_a, node_b), :]
    distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return float(distances[0])


def directed_endpoints(g: sgraph):
    endpoints = lambda branch: (int(branch[0]), int(branch[-1]))
    return [endpoints(g.path(i)) for i in range(0, g.n_paths)]


def undirected_endpoints(g: sgraph):
    return [tuple(sorted((pair))) for pair in directed_endpoints(g)]


def parallel_branches(g: sgraph):
    """
    Return node key tuples along with distances for node pairs
    with more than one branch between them.
    """
    # c = Counter(pair for pair in undirected_endpoints(g))
    # return [
    #     (*pair, distance(g, *pair))
    #     for pair, count in c.items()
    #     if count > 1
    # ]

    indices = {}
    for i in range(0, g.n_paths):
        branch = g.path(i)
        endpoints = tuple(sorted((int(branch[0]), int(branch[-1]))))
        xs = indices.setdefault(endpoints, [])
        xs.append(i)

    return [[k, v] for k, v in indices.items() if len(v) > 1]



def branch_count(g: sgraph):
    return g.n_paths


def branch_lengths(g):
    return g.path_lengths()


def node_degrees(g):
    return g.degrees


def are_branches_collinear(g: sgraph, a, b, *, atol=1e-8):
    coords = g.coordinates
    ca = coords[g.path(a)]
    cb = coords[g.path(b)]

    direction_a = ca[1] - ca[0]
    direction_b = cb[1] - cb[0]
    cross_product = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    return bool(np.isclose(cross_product, 0.0, atol=atol))


def junction_branches(g: sgraph, node):
    """
    Returns pairs of branches for a given junction node and
    whether they are collinear.
    Branch endpoints are reordered to point away from the
    junction as necessary.
    """

    branches = []
    for i, nodes in enumerate(undirected_endpoints(g)):
        if nodes[0] == nodes[1]:
            continue

        elif node in nodes:
            branches.append(i)

    pairs = []
    for i in range(0, len(branches)):
        for j in range(i + 1, len(branches)):
            pairs.append((branches[i], branches[j], are_branches_collinear(g, i, j)))

    return pairs