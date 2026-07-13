import numpy as np

def tangent(stroke):
    """
    arc-length-based; feeds curvature
    """
    tx = np.gradient(stroke.x, stroke.features["s"])
    ty = np.gradient(stroke.y, stroke.features["s"])
    return stroke.clone(features={"tx": tx, "ty": ty})
