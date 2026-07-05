import math as math
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

def resample_cubic_spline(stroke, max_points=18):
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
    n_points = max(2, min(max_points, round(total_length / math.sqrt(2) * max_points)))

    if total_length == 0:
        return np.repeat(stroke[:1], n_points, axis=0)

    sx = CubicSpline(s, stroke[:, 0], bc_type="natural")
    sy = CubicSpline(s, stroke[:, 1], bc_type="natural")
    s = np.linspace(0, total_length, n_points)
    out = np.column_stack([sx(s), sy(s)])

    # Cubic splines can overshoot slightly
    return np.clip(out, 0.0, 1.0)
