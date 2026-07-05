import numpy as np
from scipy.ndimage import gaussian_filter1d


def geometry(stroke, xy_cols=(0, 1), sigma=1.0):
    xy = stroke[:, xy_cols].astype(float)

    if len(xy) < 3:
        raise ValueError("Need at least three points.")

    result = {}

    # ----------------------------------------------------------
    # Arc length
    # ds: magnitude of vector [dxy(n), dxy(n+1)] = arc length between points n and n + 1
    # s_norm: normalized cumulative arc lengths [0, 1]
    # ----------------------------------------------------------

    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    ds = np.concatenate(([0.0], ds))
    s = np.cumsum(ds)

    # avoid duplicate arc lengths (hardly an issue for gaussian smoothed coordinates)
    s += np.arange(len(s)) * 1e-12
    s_norm = s / s[-1]

    # ----------------------------------------------------------
    # First derivative
    # ----------------------------------------------------------

    dx = np.gradient(xy[:, 0], s) # dx/ds - rate of change of the x-coordinates with respect to s
    dy = np.gradient(xy[:, 1], s) # dx/ds - rate of change of the x-coordinates with respect to s

    speed = np.hypot(dx, dy)

    tx = dx / (speed + 1e-12)
    ty = dy / (speed + 1e-12)

    tangent = np.column_stack((tx, ty))

    # ----------------------------------------------------------
    # Tangent angle
    # theta:          θ (theta); tangent angle
    # theta_gradient: change rate of θ in respect to s
    # ----------------------------------------------------------

    theta = np.unwrap(np.arctan2(ty, tx))

    # dθ/ds
    theta_gradient = np.gradient(theta, s)

    # ----------------------------------------------------------
    # Curvature
    # curvature_sd: signed curvature dT/ds
    # curvature_abs: absolute curvature dT/ds
    # ----------------------------------------------------------

    dtx = np.gradient(tx, s)
    dty = np.gradient(ty, s)

    curvature_sd = tx * dty - ty * dtx
    curvature_abs = np.abs(curvature_sd)

    # light smoothing
    # curvature_abs = gaussian_filter1d(curvature_abs, sigma=sigma)
    # curvature_sd = gaussian_filter1d(curvature_sd, sigma=sigma)
    # theta_gradient = gaussian_filter1d(theta_gradient, sigma=sigma)

    return {
        'xy': xy,
        'ds': ds,
        'norm(s)': s_norm,
        'θ': theta,
        'dθ/ds': theta_gradient,
        'K': curvature_sd,
        'abs(K)': curvature_abs
    }