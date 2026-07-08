import numpy as np

def absolute_of(sequence, point_zero):
    """
    Convert stroke sequence from relative to absolute space.
    """
    raw = sequence.numpy()
    raw = raw[:, [0, 1, 3]]
    xy = point_zero + np.cumsum(raw[:,:2], axis=0)
    return np.hstack([xy, raw[:,2:]])
