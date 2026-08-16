import numpy as np

def detect_orthogonal_persistence(stroke, min_sequence=2, step_tolerance=0.005):
    """
    Identifies indices in a stroke that exhibit 'Orthogonal Persistence'
    (quantization artifacts/staircase patterns).

    Parameters:
    - stroke: An object with .xy (np.array) and .n_points attributes.
    - min_sequence: Minimum number of alternating axial steps to flag as a staircase.
    - step_tolerance: The maximum allowed deviation from an axial unit step
      (e.g., 1.5 allows for some smoothing/rounding).

    Returns:
    - stroke object with "orthogonal_spike_indices" added to props.
    """
    points = stroke.xy
    N = stroke.n_points
    if N < min_sequence + 1:
        return stroke.clone(props={"orthogonal_spike_indices": []})

    # Calculate differences between consecutive points
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])

    # Identify if a step is "purely horizontal" or "purely vertical"
    # within the tolerance of a unit step (due to quantization).
    is_horizontal = (np.abs(dx) <= step_tolerance) & (np.abs(dy) == 0.0)
    is_vertical = (np.abs(dy) <= step_tolerance) & (np.abs(dx) == 0.0)

    spike_indices = []
    current_sequence = []

    # Iterate through the steps to find alternating axial patterns
    for i in range(len(dx)):
        if is_horizontal[i]:
            # If it's horizontal, check if previous was vertical (or start of sequence)
            if not current_sequence or is_vertical[current_sequence[-1]]:
                current_sequence.append(i)
            else:
                # Reset if we hit two horizontals in a row (not a staircase zig-zag)
                current_sequence = [i]
        elif is_vertical[i]:
            # If it's vertical, check if previous was horizontal (or start of sequence)
            if not current_sequence or is_horizontal[current_sequence[-1]]:
                current_sequence.append(i)
            else:
                # Reset if we hit two verticals in a row
                current_sequence = [i]
        else:
            # If the movement is diagonal, the staircase pattern is broken
            if len(current_sequence) >= min_sequence:
                spike_indices.extend(current_sequence)
            current_sequence = []

    # Catch sequence at the end of the stroke
    if len(current_sequence) >= min_sequence:
        spike_indices.extend(current_sequence)

    # Remove duplicates and sort (in case of overlapping logic)
    spike_indices = sorted(list(set(spike_indices)))
    staircase = np.zeros(stroke.n_points)
    staircase[spike_indices] = 1.0

    return stroke.clone(features={"staircase": staircase})