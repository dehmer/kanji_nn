import numpy as np
from . import sample_segment_adaptively

def sample_kanji_to_array(paths, error, base_density=0.1, alpha=5.0, resolution=500, epsilon=1e-5):
    """
    Samples all paths of a Kanji character into a single structured NumPy array.

    Parameters:
    -----------
    paths : list
        A list of svg.path.Path objects representing the Kanji's strokes.
    base_density : float
        Target baseline points per unit of arc length.
    alpha : float
        Sensitivity factor for curvature-dependent point clustering.
    resolution : int
        Fine grid resolution for curve analysis.
    epsilon : float
        Distance threshold below which consecutive points are considered duplicates.

    Returns:
    --------
    np.ndarray
        A 2D array of shape (N, 3) where each row is [x, y, pen_status].
        pen_status is 1.0 for writing, 0.0 for the end of a stroke.
    """
    all_stroke_points = []

    for path in paths:
        stroke_points = []

        # 1. Process all segments belonging to the current path (stroke)
        for segment in path:
            # Check for zero-length segments or structural padding
            segment_length = segment.length(error)
            if segment_length < epsilon:
                continue

            # Call your adaptive segment sampler
            sampled_seg = sample_segment_adaptively(
                segment,
                segment_length=segment_length,
                base_density=base_density,
                alpha=alpha,
                resolution=resolution
            )

            for pt in sampled_seg:
                stroke_points.append([pt['x'], pt['y']])

        if not stroke_points:
            continue

        # 2. Convert to NumPy to sanitize consecutive duplicates
        stroke_arr = np.array(stroke_points, dtype=np.float32)

        # Calculate Euclidean distances between consecutive points
        deltas = np.diff(stroke_arr, axis=0)
        distances = np.sqrt(np.sum(deltas**2, axis=1))

        # Keep the first point, and any subsequent point that actually moved
        keep_mask = np.ones(len(stroke_arr), dtype=bool)
        keep_mask[1:] = distances >= epsilon

        cleaned_stroke = stroke_arr[keep_mask]

        # Ensure the stroke still contains enough points after cleaning
        if len(cleaned_stroke) == 0:
            continue

        # 3. Append the pen feature column
        # Default all steps to 1.0 (Pen Down)
        pen_feature = np.ones((len(cleaned_stroke), 1), dtype=np.float32)
        # Mark the last coordinate of the current stroke as 0.0 (Pen Up)
        pen_feature[-1, 0] = 0.0

        # Combine [x, y] with [pen_status]
        final_stroke = np.hstack((cleaned_stroke, pen_feature))
        all_stroke_points.append(final_stroke)

    # 4. Concatenate all processed strokes into a single sequence matrix
    if not all_stroke_points:
        return np.empty((0, 3), dtype=np.float32)

    return np.vstack(all_stroke_points)
