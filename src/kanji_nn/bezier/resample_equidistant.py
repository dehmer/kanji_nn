import numpy as np
from svg.path import Path

def resample_equidistant(path: Path, n_out: int, error: float = 1e-2) -> np.ndarray:
    """
    Resamples an svg.path Path into equidistant points using a fast, non-blocking
    Lookup Table (LUT) interpolation approach.
    """
    if n_out < 2:
        raise ValueError("n_out must be at least 2.")


    fine_coords = []
    fine_seg_indices = []

    for seg_idx, seg in enumerate(path):
        # Linearly check segment type: Lines only need 2 points.
        # Curves get a safe, lightweight default (50 steps) to preserve geometry.
        seg_type = type(seg).__name__
        steps_per_segment = 2 if seg_type in ('Line', 'Close') else 50

        # Sample t linearly across this specific segment
        t_vals = np.linspace(0.0, 1.0, steps_per_segment)

        # Guard against segments that do not accept an error parameter
        try:
            pts = [seg.point(t, error=error) for t in t_vals]
        except TypeError:
            pts = [seg.point(t) for t in t_vals]

        # Convert complex numbers to [X, Y] arrays
        coords = np.array([[p.real, p.imag] for p in pts])

        # Avoid duplicating the shared node between connecting segments
        if seg_idx > 0:
            coords = coords[1:]
            indices = np.full(len(coords), seg_idx)
        else:
            indices = np.full(len(coords), seg_idx)

        fine_coords.append(coords)
        fine_seg_indices.append(indices)

    # Stack into massive flat arrays
    fine_coords = np.vstack(fine_coords)
    fine_seg_indices = np.concatenate(fine_seg_indices)

    # 2. Compute linear arc lengths between the fine sample steps
    deltas = np.diff(fine_coords, axis=0)
    step_lengths = np.sqrt(np.sum(deltas**2, axis=1))
    cum_distances = np.concatenate(([0.0], np.cumsum(step_lengths)))

    total_length = cum_distances[-1]

    # Handle zero-length paths safely
    if total_length == 0:
        out = np.zeros((n_out, 3))
        out[:, 0] = fine_coords[0, 0]
        out[:, 1] = fine_coords[0, 1]
        return out

    # 3. Linearly map perfectly even spaces to our fine distance map
    target_distances = np.linspace(0.0, total_length, n_out)

    # Interpolate X, Y coordinates and Segment Indices instantly
    new_x = np.interp(target_distances, cum_distances, fine_coords[:, 0])
    new_y = np.interp(target_distances, cum_distances, fine_coords[:, 1])

    # Use 'nearest' mapping for segment index to keep them as discrete integers
    new_seg = np.interp(target_distances, cum_distances, fine_seg_indices)
    new_seg = np.round(new_seg) # Ensure clean rounding to closest segment ID

    # 4. Construct final array
    return np.column_stack((new_x, new_y, new_seg))
