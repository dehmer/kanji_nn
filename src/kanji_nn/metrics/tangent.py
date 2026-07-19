import numpy as np

def tangent(stroke):
    """
    arc-length-based; feeds curvature
    """
    xy_smooth = stroke.features["xy_smooth"]
    s_smooth = stroke.features["s_smooth"]
    tx_raw = np.gradient(xy_smooth[:, 0], s_smooth)
    ty_raw = np.gradient(xy_smooth[:, 1], s_smooth)

    magnitude = np.sqrt(tx_raw ** 2 + ty_raw ** 2) + 1e-8
    tx = tx_raw / magnitude
    ty = ty_raw / magnitude

    return stroke.clone(features={"tx": tx, "ty": ty})
