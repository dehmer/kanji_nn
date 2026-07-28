import numpy as np

def turning_angle(stroke, w=3):
    """
    Returns (N,) array of turning angles in radians, signed,
    in (-pi, pi]. First/last w points are NaN (window doesn't
    fit on both sides).
    """

    xy = stroke.features["gauss:xy"]
    n = len(xy)
    angle = np.full(n, np.nan)

    back = xy[w:n-w] - xy[0:n-2*w]        # backward vectors
    fwd  = xy[2*w:n] - xy[w:n-w]          # forward vectors

    heading_back = np.arctan2(back[:, 1], back[:, 0])
    heading_fwd  = np.arctan2(fwd[:, 1], fwd[:, 0])

    diff = heading_fwd - heading_back
    diff = (diff + np.pi) % (2 * np.pi) - np.pi   # wrap into (-pi, pi]

    angle[w:n-w] = diff

    return stroke.clone(features={"angle": angle}, props={"w": w})