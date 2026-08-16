import numpy as np
from svg.path import Path, Move, CubicBezier
from kanji_nn.svg.classify_bezier import classify_bezier

def fit_bezier_segment(pts, p0, p3, t_hat0=None, t_hat3=None):
    """
    Least-squares fit of interior control points P1, P2 for a single
    cubic Bezier segment with FIXED endpoints p0, p3.

    pts: (n, 2) array of complex points belonging to this segment.
    p0, p3: complex numbers, the fixed segment endpoints.
    t_hat0: unit tangent leaving p0 (estimated via finite difference if omitted).
    t_hat3: unit tangent at p3, pointing BACKWARD into the curve (estimated if omitted).

    Returns: (p1, p2) as complex numbers.
    """

    d = np.abs(np.diff(pts))
    s = np.concatenate(([0.0], np.cumsum(d)))
    total = s[-1]

    if total < 1e-9:
        p1 = p0 + (p3 - p0) / 3
        p2 = p0 + 2 * (p3 - p0) / 3
        return p1, p2

    u = s / total

    tangent_window = 3
    if t_hat0 is None:
        idx_min = min(tangent_window, len(pts) - 1)
        v = pts[idx_min] - pts[0]
        t_hat0 = v / abs(v) if abs(v) > 1e-9 else (p3 - p0) / abs(p3 - p0)

    if t_hat3 is None:
        idx_max = max(len(pts) - tangent_window - 1, 0)
        v = pts[idx_max] - pts[-1]
        t_hat3 = v / abs(v) if abs(v) > 1e-9 else (p0 - p3) / abs(p0 - p3)

    B0, B1 = (1 - u) ** 3, 3 * u * (1 - u) ** 2
    B2, B3 = 3 * u ** 2 * (1 - u), u ** 3

    A0, A1 = t_hat0 * B1, t_hat3 * B2
    tmp = pts - (B0 * p0 + B3 * p3)

    c00 = np.sum((A0 * np.conj(A0)).real)
    c01 = np.sum((A0 * np.conj(A1)).real)
    c11 = np.sum((A1 * np.conj(A1)).real)
    x0 = np.sum((A0 * np.conj(tmp)).real)
    x1 = np.sum((A1 * np.conj(tmp)).real)

    det = c00 * c11 - c01 * c01
    chord_len = abs(p3 - p0)
    max_handle_len = 0.6 * chord_len

    # default: 1/3 chord length:
    alphas = [chord_len / 3] * 2
    p1 = p0 + alphas[0] * t_hat0
    p2 = p3 + alphas[1] * t_hat3

    if abs(det) < 1e-9:
        return p1, p2

    alphas[0] = (x0 * c11 - x1 * c01) / det
    alphas[1] = (c00 * x1 - c01 * x0) / det

    for alpha in alphas:
        if alpha <= 1e-6: return p1, p2            # too short
        elif alpha > max_handle_len: return p1, p2 # too long

    p1 = p0 + alphas[0] * t_hat0
    p2 = p3 + alphas[1] * t_hat3
    return p1, p2


def fit_segments(stroke):
    seg_pts = stroke.props["segments"]
    path = stroke.sticky["path"]
    n = len(seg_pts)

    segments = [Move(seg_pts[0][0])]
    for i, pts in enumerate(seg_pts):
        # p0: first of this segment
        # p3: first of next segment if any or last of this segment
        p0 = pts[0]
        p3 = seg_pts[i + 1][0] if i < n - 1 else pts[-1]
        p1, p2 = fit_bezier_segment(pts, p0, p3)
        segments.append(CubicBezier(p0, p1, p2, p3))

    path = Path(*segments)
    return stroke.clone(props={"fitted": path})
