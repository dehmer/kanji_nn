import numpy as np
from scipy.ndimage import gaussian_filter1d

def gauss_1d(stroke, sigma=1.0, mode="nearest", f=None):
    xy = stroke.features["xy"]
    gauss_xy = gaussian_filter1d(xy, axis=0, sigma=sigma, mode=mode)

    # Compensate for "shortening" of stroke.
    # Use f = 0.0 for no extrapolation.
    # Push P0 outward away from P1 and Pn outward away from Pn-1:
    # Automatically calculate the correction factor f
    f = max(0.0, 0.8 * sigma - 0.28) if f == None else f

    gauss_xy[0]  = gauss_xy[0]  + f * (gauss_xy[0]  - gauss_xy[1])
    gauss_xy[-1] = gauss_xy[-1] + f * (gauss_xy[-1] - gauss_xy[-2])

    return stroke.clone(features={"gauss:xy": gauss_xy})
