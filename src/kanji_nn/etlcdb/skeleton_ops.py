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


def _straight_segment_bounds(edt, coords, tol=1.0):
    # --- Step 1: Skip the EDT radius at the junction ---
    # The path starts at the junction, so coords[0] is our junction pixel
    junction_edt = edt[tuple(coords[0].T)]

    # Calculate cumulative distance from the start of the path
    arc_lengths = _arc_length(coords)

    # path completely lies with radius?
    if arc_lengths[-1] < junction_edt:
        return (0, len(coords))

    # Find the first index that moves past the junction's EDT radius
    # (arc_lengths is sorted so np.argmax is the way to go)
    start_idx = int(np.argmax(arc_lengths >= junction_edt))

    # --- Step 2: Find the end_idx before a significant bend ---
    # Ensure we have enough points left to define a direction vector
    if start_idx >= len(coords) - 1:
        return (0, len(coords))

    # Define the base point and a look-ahead direction vector
    # Look ahead 2-3 pixels if possible to smooth out single-pixel discretization noise
    look_ahead = min(start_idx + 3, len(coords) - 1)
    p0 = coords[start_idx]
    p1 = coords[look_ahead]

    v = p1 - p0
    v_norm = np.linalg.norm(v)

    # If the look-ahead vector has 0 length, default to the rest of the path
    if v_norm < 1e-5:
        return (start_idx, len(coords))

    v_unit = v / v_norm

    # Check all points from start_idx onwards
    remaining_coords = coords[start_idx:]

    # Vectors from p0 to each subsequent point
    vectors = remaining_coords - p0

    # Perpendicular distance: ||v x w|| / ||v||
    # In 2D, cross product of v_unit=(vx, vy) and w=(wx, wy) is |vx*wy - vy*wx|
    cross_products = vectors[:, 0] * v_unit[1] - vectors[:, 1] * v_unit[0]
    perp_distances = np.abs(cross_products)

    # Find the first index where the perpendicular distance exceeds our tolerance
    bends = perp_distances > tol
    if np.any(bends):
        # np.argmax returns the first True index relative to remaining_coords
        end_idx = start_idx + int(np.argmax(bends))
    else:
        end_idx = len(coords)

    # Ensure we return a valid slice (end_idx must be strictly greater than start_idx)
    end_idx = max(end_idx, start_idx + 1)

    return (start_idx, end_idx)


def t_junctions(glyph):
    mask = glyph[MASK].copy()
    skeleton = Skeleton(mask)
    junctions = set(np.where(skeleton.degrees == 3)[0])

    # Collect paths per junction; ensure outgoing pixel order.
    # triplets :: {int: [[int]]}
    triplets = defaultdict(list)
    for _, path in _paths(skeleton):
        if path[0] in junctions:
            triplets[path[0]].append(path)
        if path[-1] in junctions:
            triplets[path[-1]].append(path[::-1])

    return triplets


def dissolve_t_junctions(glyph):
    mask = glyph[MASK].copy()
    skeleton = Skeleton(mask)
    coords = skeleton.coordinates
    edt = glyph[EDT]

    triplets = t_junctions(glyph)

    def segment(path):
        path_coords = coords[path]
        start_idx, end_idx = _straight_segment_bounds(edt, path_coords)
        return path_coords[start_idx:end_idx]

    # junction :: int - pixel index
    # paths :: [[int]] - 3 x pixel indices
    for junction, paths in triplets.items():

        # straight-ish segment of path:
        # 1. skip edt[junciton] arc length in outward direction
        # 2. truncate before "significant" bend if any
        for i, path in enumerate(paths):
            path_coords = coords[path]
            segment_bounds = _straight_segment_bounds(edt, path_coords)
            print(junction, i, len(path), segment_bounds)
            pass

        segments = [segment(path) for path in paths]
        print(segments)

    return glyph | {MASK: mask}
