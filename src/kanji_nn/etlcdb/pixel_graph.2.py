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

        coords = self.coords[junctions]
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        n = len(junctions)

        adj = {
            i: np.where((distances[i] <= radius) & (np.arange(n) != i))[0].tolist()
            for i in range(n)
        }

        # Find connected components (clusters) using BFS:
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]: continue

            cluster = []
            queue = [i]
            visited[i] = True

            while queue:
                current = queue.pop(0)
                # Append the actual node ID from the skeleton
                cluster.append(int(junctions[current]))

                for neighbor in adj[current]:
                    if visited[neighbor]: continue
                    visited[neighbor] = True
                    queue.append(neighbor)

            clusters.append(cluster)

        return [c for c in clusters if len(c) > 1]
