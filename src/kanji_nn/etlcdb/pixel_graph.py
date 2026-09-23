import numpy as np
from skan.csr import Skeleton
from collections import defaultdict, deque


class PixelGraph:
    def __init__(self, s: Skeleton, edt):
        self.branches = [s.path(i) for i in range(0, s.n_paths)]

        # pixel index -> list of branch indices it belongs to.
        # Interior path pixels map to exactly one branch; junction/endpoint
        # pixels shared between branches map to as many branches as meet there.
        self.pixel_branches = defaultdict(list)
        for branch_idx, pixel_indices in enumerate(self.branches):
            for pixel_idx in pixel_indices:
                self.pixel_branches[int(pixel_idx)].append(branch_idx)

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

    def junction_clusters(self, radius=None):
        junctions = set(map(int, self.junctions()))

        # Deriving radius purely on the fly from skeletal path pixels
        # (Using degrees == 2 correctly samples path interiors)
        if radius is None:
            radius = 2 * np.median(self.edt[self.degrees == 2])

        adjacency = defaultdict(set)

        endpoints = zip(self.undirected_endpoints(), self.lengths)
        for (a, b), length in endpoints:
            a, b = int(a), int(b)  # Clean conversion to standard Python
            if (
                a != b
                and a in junctions
                and b in junctions
                and length <= radius
            ):
                adjacency[a].add(b)
                adjacency[b].add(a)

        visited = set()
        clusters = []

        for current in junctions:
            if current in visited:
                continue

            cluster = []
            queue = deque([current])
            visited.add(current)

            while queue:
                node = queue.popleft()
                cluster.append(node)

                for neighbour in adjacency[node]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)

            clusters.append(sorted(cluster))

        return clusters