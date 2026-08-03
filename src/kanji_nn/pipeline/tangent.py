import numpy as np

def tangent(stroke):
    """
    Calculate the normalized tangent vector components for a smoothed stroke.

    This function computes the unit tangent vector (tx, ty) at each point along
    the stroke by taking the gradient of the smoothed spatial coordinates (x, y)
    with respect to the smoothed arc length (s). A small epsilon is added to
    the magnitude to prevent division by zero during normalization.
    """
    xy_smooth = stroke.features["gauss:xy"]
    s_smooth = stroke.features["gauss:s"]
    tx_grad = np.gradient(xy_smooth[:, 0], s_smooth)
    ty_grad = np.gradient(xy_smooth[:, 1], s_smooth)

    magnitude = np.sqrt(tx_grad ** 2 + ty_grad ** 2) + 1e-8
    tx = tx_grad / magnitude
    ty = ty_grad / magnitude
    txy = np.column_stack([tx, ty])

    return stroke.clone(features={"gauss:tx": tx, "gauss:ty": ty, "gauss:txy": txy})
