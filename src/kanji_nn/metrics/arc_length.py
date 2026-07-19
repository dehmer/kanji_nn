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


    xy_smooth = gaussian_filter1d(stroke.xy, axis=0, sigma=sigma, mode=mode)
    ds_smooth = np.linalg.norm(np.diff(xy_smooth, axis=0), axis=1)
    ds_smooth = np.concatenate(([0.0], ds_smooth))
    s_smooth = np.cumsum(ds_smooth)
    s_smooth += np.arange(len(s_smooth)) * 1e-12
    s_smooth_norm = s_smooth / s_smooth[-1]


    return stroke.clone(features={
        "ds": ds,
        "s": s,
        "s_norm": s_norm,
        "xy_smooth": xy_smooth,
        "ds_smooth": ds_smooth,
        "s_smooth": s_smooth,
        "s_smooth_norm": s_smooth_norm,
    })
