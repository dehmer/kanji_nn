import numpy as np
from scipy.signal import savgol_filter

def central_speed(stroke):
    """
    Calculates central-gradient speed at each point.
    Naturally returns an N-element array.
    """
    dx_dt = np.gradient(stroke.x, stroke.t)
    dy_dt = np.gradient(stroke.y, stroke.t)

    speed = np.hypot(dx_dt, dy_dt)
    speed_sg = savgol_filter(speed, window_length=21, polyorder=3)
    speed_ds = np.gradient(speed_sg)
    speed_ds = savgol_filter(speed_ds, window_length=21, polyorder=3)
    speed_dds = np.gradient(speed_ds)

    return stroke.clone(features={
        "raw:speed:central": speed,
        "raw:speed:SG": speed_sg,
        "raw:speed:ds": speed_ds,
        "raw:speed:dds": speed_dds,
    })
