import numpy as np
from scipy.ndimage import gaussian_filter1d

def arc_length_raw(stroke, epsilon = 1e-12):
    """
    Arc length
    ds: magnitude of vector [dxy(n), dxy(n+1)] = arc length between points n and n + 1
    s_norm: normalized cumulative arc lengths [0, 1]
    """

    ds = np.linalg.norm(np.diff(stroke.xy, axis=0), axis=1)
    ds = np.concatenate(([0.0], ds))
    s = np.cumsum(ds)

    # Avoid duplicate arc lengths: s += [0 * 1e-12, 1 * 1e-12, ..., (n-1) * 1 * 1e-12]
    s += np.arange(len(s)) * epsilon
    s_norm = s / s[-1]

    return stroke.clone(features={
        "raw:ds": ds,
        "raw:s": s,
        "raw:s:norm": s_norm,
    })
