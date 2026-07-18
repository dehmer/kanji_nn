import numpy as np
from . import compute_combined_signal
from . import find_change_point

# 3. Supervised Weight Optimization Loop
def optimize_weights(strokes):
    """
    Performs a 3D grid search to find weights that minimize sample-distance MAE.
    labeled_dataset: list of dicts with stroke arrays and an 'exact_boundary_idx'
    """
    best_mae = float('inf')
    best_weights = None

    # 0.1 step resolution for weights
    grid = np.linspace(0, 1, 11)

    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-6:
                continue
            w3 = max(0.0, w3) # Sanitize floating-point errors

            total_error = 0
            for stroke in strokes:
                # Generate custom combined signal for this weight triplet
                S = compute_combined_signal(stroke, w1, w2, w3)
                # Run the sweep
                predicted_idx = find_change_point(S, window_size=6)
                # Track error

                tail_cut = stroke.props["cuts"][1]
                total_error += abs(predicted_idx - tail_cut)

            mae = total_error / len(strokes)

            if mae < best_mae:
                best_mae = mae
                best_weights = (round(w1, 2), round(w2, 2), round(w3, 2))

    return best_weights, best_mae
