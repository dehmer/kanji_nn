import numpy as np

def curvature(stroke):
    """
    # Tangent angle
    # theta:          θ (theta); tangent angle
    # theta_gradient: change rate of θ in respect to s

    # Curvature
    # curvature: signed curvature dT/ds
    # curvature_abs: absolute curvature dT/ds
    """
    tx = stroke.features["tx"]
    ty = stroke.features["ty"]
    s = stroke.features["s"]

    theta = np.unwrap(np.arctan2(ty, tx))
    theta_gradient = np.gradient(theta, s)

    dtx = np.gradient(tx, s)
    dty = np.gradient(ty, s)
    curvature = tx * dty - ty * dtx

    return stroke.clone(features={"θ": theta, "dθ/ds": theta_gradient, "K": curvature})
