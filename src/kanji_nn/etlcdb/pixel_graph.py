import numpy as np
from skan.csr import Skeleton


class PixelGraph:
    def __init__(self, s: Skeleton, edt):
        self.branches = [s.path(i) for i in range(0, s.n_paths)]
        # Flip row/column layout to x/y coordinate layout:
        self.coords = np.column_stack([s.coordinates[:, 1], s.coordinates[:, 0]])
        self.degrees = s.degrees
        self.lengths = s.path_lengths()
        self.edt = edt

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
        indices = {}
        for i, branch in enumerate(self.branches):
            endpoints = tuple(sorted((int(branch[0]), int(branch[-1]))))
            xs = indices.setdefault(endpoints, [])
            xs.append(i)

        return [[k, v] for k, v in indices.items() if len(v) > 1]
