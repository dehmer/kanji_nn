import numpy as np

def arc_length(stroke):
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

    return stroke.clone(features={
        "s": s,
        "s_norm": s_norm
    })
