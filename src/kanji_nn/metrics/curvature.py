import numpy as np

def curvature(stroke):
    """
    # Tangent angle
    # theta:          θ (theta); tangent angle
    # theta_gradient: change rate of θ in respect to s

    # Curvature
    # curvature: signed curvature dT/ds
    """
    tx = stroke.features["gauss:tx"]
    ty = stroke.features["gauss:ty"]
    s_smooth = stroke.features["gauss:s"]

    theta = np.unwrap(np.arctan2(ty, tx))
    theta_gradient = np.gradient(theta, s_smooth)

    dtx = np.gradient(tx, s_smooth)
    dty = np.gradient(ty, s_smooth)
    curvature = tx * dty - ty * dtx

    return stroke.clone(features={
        "gauss:θ": theta,
        "gauss:dθ/ds": theta_gradient,
        "gauss:K": curvature
    })
