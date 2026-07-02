import numpy as np
from scipy.interpolate import CubicSpline

def resample_cubic_spline(stroke, n_points=32):
    """
    Resample a stroke using cubic splines and arc-length parameterization.

    Parameters
    ----------
    stroke : (N,2) ndarray
        Stroke coordinates normalized to [0,1].
    n_points : int
        Number of output samples.

    Returns
    -------
    (n_points,2) ndarray
        Resampled stroke with coordinates clipped to [0,1].
    """
    stroke = np.asarray(stroke, dtype=float)

    if len(stroke) == 0:
        return np.empty((0, 2))

    if len(stroke) == 1:
        return np.repeat(stroke, n_points, axis=0)

    # Arc-length parameter
    d = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])

    total_length = s[-1]

    if total_length == 0:
        return np.repeat(stroke[:1], n_points, axis=0)

    # Remove duplicate arc-length values
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    s = s[keep]
    pts = stroke[keep]

    # If too few distinct points remain
    if len(pts) == 1:
        return np.repeat(pts, n_points, axis=0)

    # Choose spline order
    bc = "natural"

    sx = CubicSpline(s, pts[:, 0], bc_type=bc)
    sy = CubicSpline(s, pts[:, 1], bc_type=bc)

    s_new = np.linspace(0, total_length, n_points)

    out = np.column_stack([sx(s_new), sy(s_new)])

    # Cubic splines can overshoot slightly
    return np.clip(out, 0.0, 1.0)
