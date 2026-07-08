"""
Layer 1 (absolute-coordinate) augmentation: whole-character affine transforms.

transform_absolute :: stroke[] -> stroke[]

Applies ONE randomly-sampled affine transform (rotation, anisotropic scale,
shear) to ALL strokes of a character identically, via homogeneous coordinates.
Because the same matrix is applied to every stroke, relative geometry between
strokes is preserved exactly -- this is the "safe" whole-character tier of
augmentation, as opposed to per-stroke perturbation (deliberately not
implemented here).

Note: a translation component is included in build_affine_matrix() for
structural completeness, but has NO effect on the final Δx/Δy/s/pen tensor,
since differencing cancels any constant offset. Left in at 0 by default;
harmless if you ever set it, just don't expect it to visibly do anything
downstream of _character_to_tensor.
"""

import numpy as np


def build_affine_matrix(rotation=0.0, scale=(1.0, 1.0), shear=(0.0, 0.0), translation=(0.0, 0.0)):
    """
    Build a 3x3 homogeneous affine transform matrix for 2D column vectors.

    rotation    : radians
    scale       : (sx, sy)
    shear       : (shx, shy)
    translation : (tx, ty)  -- structurally present, functionally inert here (see module docstring)

    Composition order (right to left, applied to column vectors):
        M = T @ Shear @ Rotation @ Scale
    i.e. scale first, then rotate, then shear, then translate.
    """
    cos_a, sin_a = np.cos(rotation), np.sin(rotation)
    sx, sy = scale
    shx, shy = shear
    tx, ty = translation

    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1]])

    R = np.array([[cos_a, -sin_a, 0],
                  [sin_a,  cos_a, 0],
                  [0,      0,     1]])

    SH = np.array([[1,   shx, 0],
                   [shy, 1,   0],
                   [0,   0,   1]])

    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]])

    return T @ SH @ R @ S


def apply_affine(strokes, matrix):
    """
    Apply a single 3x3 homogeneous affine matrix to every stroke.

    strokes : list of (N_i, 2) arrays of absolute (x, y) points
    matrix  : (3, 3) homogeneous affine matrix

    Returns a NEW list of (N_i, 2) arrays; input strokes are left untouched.
    """
    transformed = []
    for stroke in strokes:
        n = len(stroke)
        homogeneous = np.hstack([stroke, np.ones((n, 1))])   # (N_i, 3)
        result = homogeneous @ matrix.T                      # (N_i, 3)
        transformed.append(result[:, :2])
    return transformed


# TODO: refactor
ROTATION = 10
SCALE = 0.15
SHEAR = 0.10
def transform_absolute(
    strokes,
    rng=None,
    rotation_range_deg=(-ROTATION, ROTATION),
    scale_x_range=(1.0 - SCALE, 1.0 + SCALE),
    scale_y_range=(1.0 - SCALE, 1.0 + SCALE),
    shear_range=(-SHEAR, SHEAR),
):
    """
    Apply ONE randomly-sampled whole-character affine transform to all
    strokes at once (rotation, anisotropic scale, shear -- no translation,
    see module docstring).

    strokes : list of (N_i, 2) arrays of absolute (x, y) points
    rng     : numpy.random.Generator, or None to create a fresh default one
              (pass your own if you want reproducible/seeded augmentation)

    Returns a NEW list of stroke arrays; input is left untouched.
    Designed to be injected into Dataset.__getitem__ before tensor conversion.
    """
    rng = rng or np.random.default_rng()

    rotation = np.deg2rad(rng.uniform(*rotation_range_deg))
    sx = rng.uniform(*scale_x_range)
    sy = rng.uniform(*scale_y_range)
    shx = rng.uniform(*shear_range)
    shy = rng.uniform(*shear_range)

    matrix = build_affine_matrix(rotation=rotation, scale=(sx, sy), shear=(shx, shy))
    return apply_affine(strokes, matrix)