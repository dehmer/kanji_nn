import numpy as np

def central_speed(stroke):
    """
    Calculates central-gradient speed at each point.
    Naturally returns an N-element array.
    """
    dx_dt = np.gradient(stroke.x, stroke.t)
    dy_dt = np.gradient(stroke.y, stroke.t)
    return stroke.clone(features={"c_speed": np.hypot(dx_dt, dy_dt)})
