import numpy as np
from collections import defaultdict, deque
from skan.csr import Skeleton


MASK = "skeleton:mask"
EDT = "skeleton:edt"


def _paths(skeleton, predicate=lambda index: True):
    return [
        (index, skeleton.path(index))
        for index in range(0, skeleton.n_paths)
        if predicate(index)
    ]


def _undirected_endpoints(paths):
    endpoints = lambda path: tuple(sorted([int(path[0]), int(path[-1])]))
    return [(index, endpoints(path)) for index, path in paths]


def _path_length(coords):
    return float(np.hypot(*np.diff(coords, axis=0).T).sum())


def _arc_length(coords):
    ds = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    ds = np.concatenate(([0.0], ds))
    return np.cumsum(ds)

def prune_parallel_paths(glyph):
    mask = glyph[MASK].copy()
    skeleton = Skeleton(mask)
    coords = skeleton.coordinates
    degrees = skeleton.degrees

    # Length of candidate path must not exceed this threshold:
    edt = glyph[EDT]
    pixel_edt = edt[tuple(coords.T)]
    inner_edt = np.median(pixel_edt[degrees == 2])
    radius = inner_edt + 2 * np.sqrt(inner_edt) + 1e-5

    path_lengths = skeleton.path_lengths()
    paths = _paths(skeleton, lambda index: path_lengths[index] <= radius)

    # Collect path indices per (undirected) endpoints.
    path_indices = {}
    for path_index, endpoints in _undirected_endpoints(paths):
        xs = path_indices.setdefault(endpoints, [])
        xs.append(path_index)

    parallel_paths = [
        [endpoints, indices]
        for endpoints, indices in path_indices.items()
        # ignore more than two parallel paths
        if len(indices) == 2
    ]

    if len(parallel_paths) == 0:
        return glyph

    for endpoints, paths in parallel_paths:
        lenghts = path_lengths[paths]
        sorted_paths = sorted(paths, key=lambda index: path_lengths[index])
        removals = skeleton.coordinates[skeleton.path(sorted_paths[-1])]
        for pixel in removals[1:-1]: # keep endpoints
            mask[pixel[0], pixel[1]] = False

    return prune_parallel_paths(glyph | {MASK: mask})


def dissolve_t_junctions(glyph):
    mask = glyph[MASK].copy()
    skeleton = Skeleton(mask)
    coords = skeleton.coordinates
    junctions = set(np.where(skeleton.degrees == 3)[0])
    edt = glyph[EDT]

    # Collect paths per junction; ensure outgoing pixel order.
    triplets = defaultdict(list)
    for _, path in _paths(skeleton):
        if path[0] in junctions:
            triplets[path[0]].append(path)
        if path[-1] in junctions:
            triplets[path[-1]].append(path[::-1])

    # junction :: int (pixel index)
    for junction, paths in triplets.items():
        print("junction", junction, coords[junction])
        # for i, path in enumerate(paths):
        #     path_coords = coords[path]
        #     tangent = _tangent(edt, path_coords)
        #     print(i, tangent)


    return glyph | {MASK: mask}