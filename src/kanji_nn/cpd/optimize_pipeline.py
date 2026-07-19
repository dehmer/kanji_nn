import numpy as np
from . import compute_combined_signal
from . import find_change_point

# 3. 4D Joint Optimization Loop (Weights + Adaptive Window Scale)
def optimize_pipeline(strokes, fraction=(1 - 0.15)):
    """
    Optimizes both the feature weights and the adaptive window percentage.
    """
    best_mae = float('inf')
    best_params = {}

    weight_grid = np.linspace(0, 1, 11)
    # Search from 4% to 14% of stroke length for the optimal window scale
    pct_grid = np.linspace(0.04, 0.14, 26)

    for w1 in weight_grid:
        for w2 in weight_grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-6:
                continue
            w3 = max(0.0, w3)

            for pct in pct_grid:
                total_error = 0
                for stroke in strokes:
                    # Run CPD with current adaptive window percentage
                    S = stroke.features['S']
                    predicted_idx = find_change_point(S, fraction, window_pct=pct)
                    tail_cut = stroke.props["cuts"][1]
                    total_error += abs(predicted_idx - tail_cut)

                mae = total_error / len(strokes)

                if mae < best_mae:
                    best_mae = mae
                    best_params = {
                        'weights': (round(w1, 2), round(w2, 2), round(w3, 2)),
                        'window_pct': round(pct, 3)
                    }

    return best_params, best_mae
