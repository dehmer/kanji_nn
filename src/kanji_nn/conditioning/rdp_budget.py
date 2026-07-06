"""
Binary-search-to-budget RDP simplification for KanjiVG stroke data.

Input: list of strokes, each a (N_i, 2) numpy array of normalized [0,1] points
       (this is exactly what you get back from reading a WKB MultiLineString
       via shapely: [np.asarray(ls.coords) for ls in geom.geoms])

Output: simplified list of strokes, same structure, total point count <= target budget.

Two strategies:
  1. rdp_to_budget_flat   - one global epsilon for the whole character (binary search)
  2. rdp_to_budget_weighted - per-stroke budget proportional to arc length,
                              each stroke gets its own binary-searched epsilon
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core RDP
# ---------------------------------------------------------------------------

def _perp_distances(points, start, end):
    """Perpendicular distance of each point to the line (start, end)."""
    if np.allclose(start, end):
        return np.linalg.norm(points - start, axis=1)
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    vec_from_start = points - start
    proj_len = vec_from_start @ line_unit
    proj_point = start + np.outer(proj_len, line_unit)
    return np.linalg.norm(points - proj_point, axis=1)


def rdp(points, epsilon):
    """
    Ramer-Douglas-Peucker simplification.
    points: (N, 2) array. Always keeps first and last point.
    Returns a (M, 2) array, M <= N.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3:
        return points

    # iterative stack-based RDP to avoid recursion-depth issues on long strokes
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]

    while stack:
        start_idx, end_idx = stack.pop()
        if end_idx - start_idx < 2:
            continue
        segment = points[start_idx + 1:end_idx]
        dists = _perp_distances(segment, points[start_idx], points[end_idx])
        max_i = np.argmax(dists)
        max_dist = dists[max_i]
        if max_dist > epsilon:
            split_idx = start_idx + 1 + max_i
            keep[split_idx] = True
            stack.append((start_idx, split_idx))
            stack.append((split_idx, end_idx))

    return points[keep]


# ---------------------------------------------------------------------------
# Strategy 1: flat global epsilon, binary search on total point count
# ---------------------------------------------------------------------------

def rdp_to_budget_flat(strokes, target_total, eps_lo=1e-5, eps_hi=0.1,
                        iters=20, tol_points=0):
    """
    Single epsilon applied to every stroke, binary-searched so the total
    point count across all strokes is as close as possible to target_total
    without exceeding it.

    strokes: list of (N_i, 2) arrays
    target_total: desired total point count (e.g. 256, minus any bookkeeping
                  you plan to add later such as pen-up duplicates)
    Returns: (simplified_strokes, eps_used, total_points)
    """
    best = None
    best_eps = eps_hi

    for _ in range(iters):
        eps = (eps_lo + eps_hi) / 2
        simplified = [rdp(s, eps) for s in strokes]
        total = sum(len(s) for s in simplified)

        if total > target_total:
            eps_lo = eps
        else:
            eps_hi = eps
            # track the tightest fit that still satisfies the budget
            if best is None or total > sum(len(s) for s in best):
                best = simplified
                best_eps = eps

        if abs(total - target_total) <= tol_points:
            best = simplified
            best_eps = eps
            break

    if best is None:
        # even eps_hi couldn't get under budget (pathological / eps_hi too small)
        best = [rdp(s, eps_hi) for s in strokes]
        best_eps = eps_hi

    return best, best_eps, sum(len(s) for s in best)


# ---------------------------------------------------------------------------
# Strategy 2: per-stroke budget weighted by arc length, independent
#             binary search per stroke
# ---------------------------------------------------------------------------

def _arc_length(points):
    if len(points) < 2:
        return 0.0
    return np.linalg.norm(np.diff(points, axis=0), axis=1).sum()


def _rdp_to_point_count(points, target_n, eps_lo=1e-5, eps_hi=0.1, iters=15):
    """Binary search epsilon for a single stroke to hit <= target_n points."""
    points = np.asarray(points, dtype=float)
    if len(points) <= max(target_n, 2):
        return points  # already within budget, nothing to do

    best = rdp(points, eps_hi)
    for _ in range(iters):
        eps = (eps_lo + eps_hi) / 2
        simplified = rdp(points, eps)
        if len(simplified) > target_n:
            eps_lo = eps
        else:
            eps_hi = eps
            best = simplified
    return best


def rdp_to_budget_weighted(strokes, target_total, min_points_per_stroke=2):
    """
    Allocates each stroke a point budget proportional to its share of total
    arc length, then simplifies each stroke independently to its sub-budget.

    Every stroke is guaranteed at least min_points_per_stroke (default 2:
    just the endpoints), so short strokes never vanish and long/curvy
    strokes get more of the 180/240/256 point budget than a flat scheme
    would give them.

    Returns: (simplified_strokes, per_stroke_targets, total_points)
    """
    n_strokes = len(strokes)
    lengths = np.array([_arc_length(s) for s in strokes])
    total_length = lengths.sum()

    # reserve the mandatory minimum for every stroke first
    reserved = n_strokes * min_points_per_stroke
    remaining_budget = max(target_total - reserved, 0)

    if total_length == 0:
        # degenerate: all zero-length strokes (shouldn't happen in practice)
        weights = np.ones(n_strokes) / n_strokes
    else:
        weights = lengths / total_length

    extra_per_stroke = np.floor(weights * remaining_budget).astype(int)
    per_stroke_targets = extra_per_stroke + min_points_per_stroke

    # distribute any leftover points (from flooring) to the longest strokes
    leftover = target_total - per_stroke_targets.sum()
    if leftover > 0:
        order = np.argsort(-lengths)
        for i in range(leftover):
            per_stroke_targets[order[i % n_strokes]] += 1

    simplified = [
        _rdp_to_point_count(s, int(t))
        for s, t in zip(strokes, per_stroke_targets)
    ]

    return simplified, per_stroke_targets, sum(len(s) for s in simplified)


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     rng = np.random.default_rng(0)

#     def fake_stroke(n, curviness=0.02):
#         t = np.linspace(0, 1, n)
#         x = t
#         y = 0.3 * np.sin(t * np.pi * 2) + curviness * rng.standard_normal(n)
#         return np.stack([x, y], axis=1)

#     # simulate a 12-stroke character with mixed complexity
#     strokes = [fake_stroke(rng.integers(15, 60)) for _ in range(12)]
#     original_total = sum(len(s) for s in strokes)
#     print(f"original total points: {original_total}")

#     flat_simplified, eps_used, flat_total = rdp_to_budget_flat(strokes, target_total=256)
#     print(f"[flat]     eps={eps_used:.5f}  total={flat_total}  "
#           f"per-stroke={[len(s) for s in flat_simplified]}")

#     weighted_simplified, targets, weighted_total = rdp_to_budget_weighted(strokes, target_total=256)
#     print(f"[weighted] total={weighted_total}  targets={list(targets)}  "
#           f"actual={[len(s) for s in weighted_simplified]}")