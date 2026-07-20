import numpy as np

def detect_hooks(
    geometry,
    # endpoint_fraction=0.15,
    # curvature_factor=2.5,
    # min_turn_deg=25,
    # baseline_window=5,
):
    # Looking at fixed fractions at start/end won't cut it.
    # Fractions might be too short for short strokes.
    # Idea: We sweep from the left and right to find a 'stable' section.

    s = geometry["norm(s)"]
    n = len(s)
    gradient = np.abs(geometry["gauss:dθ/ds"])

    condition = gradient > 40
    padded = np.empty(len(condition) + 2, dtype=bool)
    padded[0], padded[-1] = False, False
    padded[1:-1] = condition
    transitions = np.diff(padded.view(np.int8))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    ranges = list(zip(starts, ends))

    start = 0 # inclusive
    end = n   # exclusive

    for range in ranges:
        if (range[0] <= 5): start = range[1] - 1
        if (range[1] > n - 5): end = range[0] + 1

    if (start > end):
        return 0, n

    return start, end
