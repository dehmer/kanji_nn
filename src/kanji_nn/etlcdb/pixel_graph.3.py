import numpy as np
from skan.csr import Skeleton
from scipy.spatial.distance import cdist

class PixelGraph:
    def __init__(self, s: Skeleton, edt):
        self.branches = [s.path(i) for i in range(0, s.n_paths)]
        # Flip row/column layout to x/y coordinate layout:
        x, y = s.coordinates[:, 1], s.coordinates[:, 0]

        self.coords = np.column_stack([x, y])
        self.degrees = s.degrees
        self.lengths = s.path_lengths()

        # align/filter raw euclidean distance transform to skeleton pixels:
        self.edt = edt[y, x]

        # keep raw EDT matrix hanging around for possible copy constructor:
        self.edt_reference = edt

    def junctions(self):
        return np.where(self.degrees > 2)[0]

    def endpoints(self):
        return np.where(self.degrees == 1)[0]

    def distance(self, a, b):
        "Return Euclidean distance between two nodes."
        coords = self.coords[(a, b), :]
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        return float(distances[0])

    def directed_endpoints(self):
        endpoints = lambda branch: (int(branch[0]), int(branch[-1]))
        return [endpoints(branch) for branch in self.branches]

    def undirected_endpoints(self):
        return [tuple(sorted((pair))) for pair in self.directed_endpoints()]

    def parallel_branches(self):
        indices = {}
        for i, branch in enumerate(self.branches):
            endpoints = tuple(sorted((int(branch[0]), int(branch[-1]))))
            xs = indices.setdefault(endpoints, [])
            xs.append(i)

        return [[k, v] for k, v in indices.items() if len(v) > 1]

    def junction_clusters(self):
        junctions = self.junctions()
        radius = 2 * np.median(self.edt)

        edges = [
            endpoints
            for endpoints, length in zip(self.undirected_endpoints(), self.lengths)
            if endpoints[0] in junctions
            and endpoints[1] in junctions
            and endpoints[0] != endpoints[1]
            and length <= radius
        ]

        print("edges", edges)
