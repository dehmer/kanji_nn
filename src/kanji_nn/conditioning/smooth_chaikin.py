import numpy as np

def smooth_chaikin(stroke, refinements=3):
    """
    Smooths a polyline using Chaikin's corner-cutting algorithm.
    Accepts a NumPy array of shape (N, 2).
    """

    for _ in range(refinements):
        # Duplicate rows to handle the 1:3 cutting ratios
        L = stroke.repeat(2, axis=0)

        R = np.empty_like(L)

        # Shift and align the arrays to compute vector ratios
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]

        # Apply Chaikin weight formula: 1/4 and 3/4 cuts
        stroke = L * 0.75 + R * 0.25

    return stroke
