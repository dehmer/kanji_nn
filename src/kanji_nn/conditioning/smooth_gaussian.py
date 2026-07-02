import numpy as np

def smooth_gaussian(stroke):
    """
    Applies a 1D Gaussian filter (window=3) to the X and Y coordinates
    of a individual stroke. Timestamps are left completely unaltered.

    Parameters:
    strokes_list (list of numpy.ndarray): Each element shape (m, 3) -> [x, y, timestamp]

    Returns:
    list of numpy.ndarray: Smoothed strokes with identical shapes.
    """

    # Define a symmetric Gaussian kernel for window size 3
    # Weights sum up to 1.0 to preserve the spatial scale
    kernel = np.array([0.25, 0.50, 0.25], dtype=np.float32)


    column_count = stroke.shape[1]
    X, Y = (0, 1) if column_count == 3 else (1, 2)

    # If the stroke is too short to smooth, keep it as is
    if stroke.shape[0] < 3:
        return stroke.copy()

    # Clone the stroke structure to keep timestamps intact
    smoothed_stroke = stroke.copy()

    # Convolve X (col 0) and Y (col 1) independently
    # 'edge' padding prevents the stroke endpoints from shrinking inward
    smoothed_stroke[:, X] = np.convolve(stroke[:, X], kernel, mode='same')
    smoothed_stroke[:, Y] = np.convolve(stroke[:, Y], kernel, mode='same')

    # Explicitly preserve the exact starting and ending coordinate positions
    # to prevent the convolution padding from introducing artificial shifts
    smoothed_stroke[0, X:Y+1] = stroke[0, X:Y+1]
    smoothed_stroke[-1, X:Y+1] = stroke[-1, X:Y+1]
    return smoothed_stroke
