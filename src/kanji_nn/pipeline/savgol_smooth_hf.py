import numpy as np
from scipy.signal import savgol_filter


def savgol_smooth_hf(stroke, k=0.2, min_window=5, max_polyorder=2):
    """
    Smooths stroke.xy with a Savitzky-Golay filter. window_length scales
    with n_points (fraction k), floored at min_window and capped at
    n_points, always odd. polyorder is capped at max_polyorder (and at
    window - 1, to stay legal on very short strokes).

    Shortest observed stroke so far: n_points=6 -> window=5, polyorder=2.
    """
    n = stroke.n_points

    window = max(min_window, round(n * k))
    window = min(window, n)
    if window % 2 == 0:
        window -= 1
    window = max(window, 3)  # absolute floor so polyorder=2 stays legal

    polyorder = min(max_polyorder, window - 1)

    xy_smooth = savgol_filter(
        stroke.xy, window_length=window, polyorder=polyorder, axis=0, mode="interp"
    )

    return stroke.clone(features={
        "savgol:hf:xy": xy_smooth,
    }, props={
        "savgol:hf:window": window,
        "savgol:hf:polyorder": polyorder,
    })
