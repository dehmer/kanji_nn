import numpy as np
from scipy.ndimage import gaussian_filter1d

def gauss_1d(stroke, sigma=1.0, mode="reflect"):
    gauss_xy = gaussian_filter1d(stroke.xy, axis=0, sigma=sigma, mode=mode)
    features = {"gauss:xy": gauss_xy}
    props = {"sigma": sigma, "mode": mode}

    return stroke.clone(features=features, props=props)
