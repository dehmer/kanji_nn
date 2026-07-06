"""
Adaptive sampling of SVG path data (KanjiVG strokes) into point sequences.

Uses svg.path to parse the `d` attribute into segments (Line, CubicBezier,
etc.), then recursively subdivides each Bezier segment (de Casteljau) only
as much as needed to stay within a flatness tolerance. Straight-ish stretches
collapse to just their endpoints; hooks and sharp corners get subdivided
until they're well represented.

Dependencies: svg.path, numpy
    pip install svg.path numpy
"""

import numpy as np
from svg.path import parse_path
from svg.path.path import Move, Close, Line, CubicBezier, QuadraticBezier


# ---------------------------------------------------------------------------
# Geometry helpers (operate on plain (x, y) numpy pairs, not complex numbers,
# to keep the de Casteljau / flatness math easy to follow)
# ---------------------------------------------------------------------------

def _to_xy(c):
    """svg.path represents points as Python complex numbers: x + yj."""
    return np.array([c.real, c.imag], dtype=float)


def _flatness(p0, p1, p2, p3):
    """
    How far the two control points deviate from the straight chord p0->p3.
    Returns the larger of the two perpendicular distances.
    A value near 0 means the curve is already basically a straight line.
    """
    chord = p3 - p0
    chord_len = np.linalg.norm(chord)
    if chord_len < 1e-12:
        # degenerate: p0 == p3, fall back to control point spread
        return max(np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p0))
    chord_unit = chord / chord_len

    def perp_dist(p):
        proj_len = np.dot(p - p0, chord_unit)
        proj_point = p0 + proj_len * chord_unit
        return np.linalg.norm(p - proj_point)

    return max(perp_dist(p1), perp_dist(p2))


def _de_casteljau_split(p0, p1, p2, p3, t=0.5):
    """
    Split a cubic Bezier at parameter t into two cubic Beziers that together
    trace exactly the same curve. Returns (left_4_points, right_4_points).
    """
    p01 = (1 - t) * p0 + t * p1
    p12 = (1 - t) * p1 + t * p2
    p23 = (1 - t) * p2 + t * p3

    p012 = (1 - t) * p01 + t * p12
    p123 = (1 - t) * p12 + t * p23

    p0123 = (1 - t) * p012 + t * p123

    left = (p0, p01, p012, p0123)
    right = (p0123, p123, p23, p3)
    return left, right


# ---------------------------------------------------------------------------
# Adaptive flattening of a single cubic Bezier segment
# ---------------------------------------------------------------------------

def adaptive_flatten_cubic(p0, p1, p2, p3, tol=0.01, max_depth=12, _depth=0):
    """
    Recursively subdivide a cubic Bezier until each piece is flat enough
    (within `tol`), then return the sampled points along it.

    Returns a list of (x, y) numpy arrays INCLUDING p0 but EXCLUDING p3
    (the caller is responsible for stitching segments together and adding
    the very final point once, to avoid duplicate points at segment joins).
    """
    if _depth >= max_depth or _flatness(p0, p1, p2, p3) < tol:
        return [p0]

    left, right = _de_casteljau_split(p0, p1, p2, p3, 0.5)
    left_points = adaptive_flatten_cubic(*left, tol, max_depth, _depth + 1)
    right_points = adaptive_flatten_cubic(*right, tol, max_depth, _depth + 1)
    return left_points + right_points


# ---------------------------------------------------------------------------
# Full stroke path -> list of sampled points
# ---------------------------------------------------------------------------

def sample_path_d(path, tol=0.01, max_depth=12):
    """
    Parse an SVG path `d` string and adaptively sample it into a single
    polyline of (x, y) points (in the SVG's original coordinate units --
    do your /109 KanjiVG normalization afterwards).

    Handles Move, Line, CubicBezier. QuadraticBezier is elevated to cubic
    for a uniform code path. Close is treated like a Line back to the
    subpath's start point.

    A single KanjiVG stroke's `d` is normally one continuous subpath, so
    this returns one (N, 2) array. If the `d` string contains multiple
    Move commands (not typical for a single KanjiVG stroke, but handled
    defensively), only the first subpath is sampled -- call this once per
    subpath if you need to support that case.
    """
    points = []
    subpath_start = None

    for segment in path:
        if isinstance(segment, Move):
            start = _to_xy(segment.start)
            if not points:
                points.append(start)
            subpath_start = start

        elif isinstance(segment, Line):
            # straight line: just the two endpoints, no subdivision needed
            end = _to_xy(segment.end)
            points.append(end)

        elif isinstance(segment, Close):
            if subpath_start is not None:
                points.append(subpath_start)

        elif isinstance(segment, QuadraticBezier):
            # elevate quadratic -> cubic so we can reuse the same flattening code:
            # C1 = Q0 + 2/3*(Q1-Q0), C2 = Q2 + 2/3*(Q1-Q2)
            p0 = _to_xy(segment.start)
            q1 = _to_xy(segment.control)
            p3 = _to_xy(segment.end)
            p1 = p0 + (2 / 3) * (q1 - p0)
            p2 = p3 + (2 / 3) * (q1 - p3)
            seg_points = adaptive_flatten_cubic(p0, p1, p2, p3, tol, max_depth)
            points.extend(seg_points[1:] if points else seg_points)
            points.append(p3)

        elif isinstance(segment, CubicBezier):
            p0 = _to_xy(segment.start)
            p1 = _to_xy(segment.control1)
            p2 = _to_xy(segment.control2)
            p3 = _to_xy(segment.end)
            seg_points = adaptive_flatten_cubic(p0, p1, p2, p3, tol, max_depth)
            # seg_points already starts with p0; drop it if we already have
            # that point from the previous segment to avoid a duplicate at
            # the join (this is also where corners are naturally preserved,
            # since we never simplify ACROSS a segment boundary)
            points.extend(seg_points[1:] if points else seg_points)
            points.append(p3)

    return np.array(points, dtype=float)


def sample_kanjivg_char(parsed_paths, tol=0.01, max_depth=12, viewbox_size=109.0):
    """
    Convenience wrapper: sample every stroke of a character and normalize
    to [0, 1] by dividing by the KanjiVG viewBox size (109x109).

    stroke_d_list: list of `d` strings, one per <path> element, in the
                   stroke-order given by the KanjiVG file.

    Returns: list of (N_i, 2) numpy arrays, one per stroke, normalized.
    """
    strokes = []
    for path in parsed_paths:
        pts = sample_path_d(path, tol=tol, max_depth=max_depth)
        strokes.append(pts / viewbox_size)
    return strokes


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     # A single stroke shaped roughly like a hook: mostly straight, then a
#     # sharp curl at the end -- similar to the tail flick on many kanji strokes.
#     example_d = "M 20,20 C 20,20 60,20 90,25 C 95,26 40,80 30,60 C 25,50 45,40 55,45"

#     pts = sample_path_d(example_d, tol=0.5, max_depth=12)
#     print(f"sampled {len(pts)} points from a 3-segment cubic path:")
#     print(pts)

#     print()
#     print("effect of tolerance on point count:")
#     for tol in [5.0, 1.0, 0.5, 0.1, 0.01]:
#         pts = sample_path_d(example_d, tol=tol)
#         print(f"  tol={tol:<6} -> {len(pts)} points")

#     print()
#     print("normalized character example (two fake strokes):")
#     strokes = sample_kanjivg_char([
#         "M 20,20 L 90,20",                       # a straight horizontal stroke
#         "M 50,10 C 50,10 55,90 20,95",            # a curved vertical stroke
#     ], tol=0.5)
#     for i, s in enumerate(strokes):
#         print(f"  stroke {i}: {len(s)} points, range x[{s[:,0].min():.3f},{s[:,0].max():.3f}] "
#               f"y[{s[:,1].min():.3f},{s[:,1].max():.3f}]")