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

    """
    TODO: jitter = near-zero net displacement (gauss:s) = curvature blow-up

    possible fixes:

    1 Floor the denominator: clip s_smooth differences to a minimum
       epsilon before differentiating, so near-zero speed doesn't create
       unbounded ratios.

    2. Gate on speed: mask/exclude or heavily downweight K at samples
       where ds_smooth (or c_speed) is below some small threshold —
       arc-length-based curvature is only meaningful where there's
       meaningful arc length between samples.

    3. Winsorize/clip post-hoc: cap K at some percentile before it ever
       becomes an HMM feature, since a Gaussian's variance estimate gets
       wrecked by a handful of 1e11-scale outliers.
    """

    dtx = np.gradient(tx, s_smooth)
    dty = np.gradient(ty, s_smooth)
    curvature = tx * dty - ty * dtx

    return stroke.clone(features={
        "gauss:θ": theta,
        "gauss:dθ/ds": theta_gradient,
        "gauss:K": curvature
    })
