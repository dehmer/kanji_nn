import numpy as np

def tangent(stroke):
    """
    arc-length-based; feeds curvature
    """
    xy_smooth = stroke.features["gauss:xy"]
    s_smooth = stroke.features["gauss:s"]
    tx_grad = np.gradient(xy_smooth[:, 0], s_smooth)
    ty_grad = np.gradient(xy_smooth[:, 1], s_smooth)

    magnitude = np.sqrt(tx_grad ** 2 + ty_grad ** 2) + 1e-8
    tx = tx_grad / magnitude
    ty = ty_grad / magnitude

    return stroke.clone(features={"gauss:tx": tx, "gauss:ty": ty})
