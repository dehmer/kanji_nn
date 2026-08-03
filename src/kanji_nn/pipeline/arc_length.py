import numpy as np
from scipy.ndimage import gaussian_filter1d

def arc_length(stroke, sigma=1.0, mode='reflect'):
    """
    Arc length
    ds: magnitude of vector [dxy(n), dxy(n+1)] = arc length between points n and n + 1
    s_norm: normalized cumulative arc lengths [0, 1]
    """

    ds = np.linalg.norm(np.diff(stroke.xy, axis=0), axis=1)
    ds = np.concatenate(([0.0], ds))
    s = np.cumsum(ds)

    # Avoid duplicate arc lengths: s += [0 * 1e-12, 1 * 1e-12, ..., (n-1) * 1 * 1e-12]
    s += np.arange(len(s)) * 1e-12
    s_norm = s / s[-1]

    gauss_xy = gaussian_filter1d(stroke.xy, axis=0, sigma=sigma, mode=mode)
    gauss_ds = np.linalg.norm(np.diff(gauss_xy, axis=0), axis=1)
    gauss_ds = np.concatenate(([0.0], gauss_ds))
    gauss_s = np.cumsum(gauss_ds)
    gauss_s += np.arange(len(gauss_s)) * 1e-12

    return stroke.clone(features={
        "raw:ds": ds,
        "raw:s": s,
        "raw:s:norm": s_norm,
        "gauss:xy": gauss_xy,
        "gauss:ds": gauss_ds,
        "gauss:s": gauss_s,
    })
