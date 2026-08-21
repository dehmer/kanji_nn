import numpy as np

def central_speed(stroke):
    """
    Calculates central-gradient speed at each point.
    Naturally returns an N-element array.
    """
    t = stroke.features["t"]
    xy = stroke.features["xy"]
    x = xy[:, 0]
    y = xy[:, 1]
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    speed = np.hypot(dx_dt, dy_dt)

    return stroke.clone(features={
        "raw:speed:central": speed,
    })
