import numpy as np
from .typing import Splines

def _curve_point(splines: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate each of N curves at its own parameter t. splines: (N,9), t: (N,) -> (N,2)."""
    p0, p1, p2, p3 = splines[:, 0:2], splines[:, 2:4], splines[:, 4:6], splines[:, 6:8]
    t = t[:, None]
    mt = 1 - t
    return mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3


def splines_bbox(splines: Splines) -> np.ndarray:
    """Exact bbox [xmin, ymin, xmax, ymax] over an entire (N, 9) Splines array."""
    p0, p1, p2, p3 = splines[:, 0:2], splines[:, 2:4], splines[:, 4:6], splines[:, 6:8]

    a = -p0 + 3 * p1 - 3 * p2 + p3
    b = 2 * p0 - 4 * p1 + 2 * p2
    c = p1 - p0

    disc = b**2 - 4 * a * c
    quad_mask = (a != 0) & (disc >= 0)
    sqrt_disc = np.sqrt(np.where(disc >= 0, disc, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = (-b + sqrt_disc) / (2 * a)
        r2 = (-b - sqrt_disc) / (2 * a)
    r1 = np.where(quad_mask, r1, 0.0)
    r2 = np.where(quad_mask, r2, 0.0)

    lin_mask = (a == 0) & (b != 0)
    r_lin = np.where(b != 0, -c / np.where(b != 0, b, 1), 0.0)
    r1 = np.where(lin_mask, r_lin, r1)

    N = splines.shape[0]
    zeros, ones = np.zeros(N), np.ones(N)
    candidates = [zeros, ones, r1[:, 0], r2[:, 0], r1[:, 1], r2[:, 1]]
    candidates = [np.clip(t, 0.0, 1.0) for t in candidates]

    points = np.stack([_curve_point(splines, t) for t in candidates], axis=1)  # (N, 6, 2)
    mins = points.min(axis=1)  # (N, 2)
    maxs = points.max(axis=1)

    return np.array([mins[:, 0].min(), mins[:, 1].min(), maxs[:, 0].max(), maxs[:, 1].max()])
