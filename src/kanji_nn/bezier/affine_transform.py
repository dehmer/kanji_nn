import numpy as np

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

def apply_affine_transform(spline, m):
    """
    Applies a pre-composed 3x3 affine matrix to an (N, 8) array of Bezier curves.
    """
    N = spline.shape[0]

    # Reshape from (N, 8) to (N * 4, 2) to get a list of flat (X, Y) coordinate rows
    points_2d = spline.reshape(-1, 2)

    # Convert to homogeneous coordinates by appending a column of 1s -> (N * 4, 3)
    homogeneous_points = np.hstack([points_2d, np.ones((points_2d.shape[0], 1))])

    # Apply the 3x3 matrix via dot product
    # Since our coordinates are arranged as row vectors [X, Y, 1],
    # we evaluate: points . M^T
    transformed_homogeneous = homogeneous_points @ m.T

    # 4. Strip the homogeneous 1s and reshape back to original (N, 8) format
    return transformed_homogeneous[:, :2].reshape(N, 8)
