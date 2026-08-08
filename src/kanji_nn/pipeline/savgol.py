import numpy as np
from scipy.signal import savgol_filter


def savgol(stroke, window_length=5, polyorder=2, mode="interp"):
    """
    Smooths stroke.xy with a Savitzky-Golay filter.
    Shortest observed stroke so far: n_points=6 -> window=5, polyorder=2.
    """
    xy = savgol_filter(
        stroke.xy,
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
        mode=mode
    )

    return stroke.clone(features={"savgol:xy": xy})
