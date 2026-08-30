import numpy as np
from .typing import Splines

def m_translate(tx, ty):
    """Returns a 3x3 homogeneous translation matrix."""
    return np.array([
        [1.0, 0.0,  tx],
        [0.0, 1.0,  ty],
        [0.0, 0.0, 1.0]
    ])

def m_scale(sx, sy):
    """Returns a 3x3 homogeneous scaling matrix."""
    return np.array([
        [ sx, 0.0, 0.0],
        [0.0,  sy, 0.0],
        [0.0, 0.0, 1.0]
    ])

def m_rotate(deg):
    """Returns a 3x3 homogeneous rotation matrix around the origin."""
    rad = np.radians(deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    return np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [  0.0,    0.0, 1.0]
    ])

def m_shear(shx, shy):
    """Returns a 3x3 homogeneous shear matrix."""
    return np.array([
        [1.0, shx, 0.0],
        [shy, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])


def transform_splines(splines: Splines, m: np.ndarray) -> Splines:
    """
    Apply a 3x3 homogeneous matrix m to all p0..p3 points in a Splines array.
    Pen column untouched.
    """
    coords = splines[:, :8].reshape(-1, 2)                            # (N*4, 2)
    homogeneous = np.hstack([coords, np.ones((coords.shape[0], 1))])  # (N*4, 3)
    transformed = homogeneous @ m.T                                   # (N*4, 3)
    new_coords = transformed[:, :2].reshape(-1, 8)                    # (N, 8)
    return np.hstack([new_coords, splines[:, 8:9]])