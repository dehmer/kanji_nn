import numpy as np
from collections import defaultdict, deque
from skan.csr import Skeleton

MASK = "skeleton:mask"
EDT = "skeleton:edt:inner"


def _paths(skeleton, predicate=lambda index: True):
    return [
        (index, skeleton.path(index))
        for index in range(0, skeleton.n_paths)
        if predicate(index)
    ]


def _undirected_endpoints(paths):
    endpoints = lambda path: tuple(sorted([int(path[0]), int(path[-1])]))
    return [(index, endpoints(path)) for index, path in paths]


def parallel_paths(mask):
    mask = glyph[MASK]
    skeleton = Skeleton(mask)

    # Length of candidate path must not exceed this threshold:
    edt_inner = glyph[EDT]
    radius = edt_inner + 2 * np.sqrt(edt_inner) + 1e-5

    path_lengths = skeleton.path_lengths()
    paths = _paths(skeleton, lambda index: path_lengths[index] <= radius)

    # Collect path indices per (undirected) endpoints.
    path_indices = {}
    for path_index, endpoints in _undirected_endpoints(paths):
        xs = path_indices.setdefault(endpoints, [])
        xs.append(path_index)

    return [
        [endpoints, indices]
        for endpoints, indices in path_indices.items()
        # ignore more than two parallel paths
        if len(indices) == 2
    ]


def prune_parallel_paths(glyph):
    """
    """
    mask = glyph[MASK]
    skeleton = Skeleton(mask)

    # Length of candidate path must not exceed this threshold:
    edt_inner = glyph[EDT]
    radius = edt_inner + 2 * np.sqrt(edt_inner) + 1e-5

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

    for endpoints, paths in parallel_paths:
        lenghts = path_lengths[paths]
        sorted_paths = sorted(paths, key=lambda index: path_lengths[index])
        # TODO: rounded mid-point for equal length
        removals = skeleton.coordinates[skeleton.path(sorted_paths[-1])]
        for pixel in removals[1:-1]: # keep endpoints
            mask[pixel[0], pixel[1]] = False

    return glyph | {MASK: mask}
