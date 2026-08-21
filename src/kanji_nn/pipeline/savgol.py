import numpy as np
from scipy.signal import savgol_filter

# TODO: checkout mode='nearest'
def savgol(stroke, window_length=5, polyorder=2, mode="nearest"):
    """
    Smooths stroke.xy with a Savitzky-Golay filter.
    Shortest observed stroke so far: n_points=6 -> window=5, polyorder=2.
    """
    xy = stroke.features["xy"]
    xy = savgol_filter(
        xy,
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
        mode=mode
    )

    return stroke.clone(features={"savgol:xy": xy})
