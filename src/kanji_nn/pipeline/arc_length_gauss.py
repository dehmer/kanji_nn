import numpy as np
from scipy.ndimage import gaussian_filter1d

def arc_length_gauss(stroke, sigma=1.0, mode='reflect'):
    gauss_xy = gaussian_filter1d(stroke.xy, axis=0, sigma=sigma, mode=mode)
    gauss_ds = np.linalg.norm(np.diff(gauss_xy, axis=0), axis=1)
    gauss_ds = np.concatenate(([0.0], gauss_ds))
    gauss_s = np.cumsum(gauss_ds)
    gauss_s += np.arange(len(gauss_s)) * 1e-12

    return stroke.clone(features={
        "gauss:xy": gauss_xy,
        "gauss:ds": gauss_ds,
        "gauss:s": gauss_s,
    })
