import numpy as np

def backward_speed(stroke):
    """
    Calculates backward-difference speed over segments.
    Pads with a 0.0 at the beginning to return an N-element array.
    """
    dt = np.diff(stroke.t)
    dxy = np.diff(stroke.xy, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate raw interval speeds (N-1 elements)
        speed = np.where(dt > 0, np.hypot(dxy[:, 0], dxy[:, 1]) / dt, 0.0)

    # Right-pad with a single 0.0 to restore size to N elements
    speed =  np.pad(speed, (1, 0), mode='constant', constant_values=0.0)
    return stroke.clone(features={"backward_speed": speed})
