import numpy as np
import itertools
from . import find_change_point


# 3. 4D Joint Optimization Loop (Weights + Adaptive Window Scale)
def optimize_pipeline(strokes, S, n, fraction=(1 - 0.15)):
    """
    Optimizes both the feature weights and the adaptive window percentage.
    """

    best_mae = float('inf')
    best_params = {}

    grid_size = 21
    weight_grid = np.linspace(0, 1, grid_size)

    # Search from 4% to 14% of stroke length for the optimal window scale
    pct_grid = np.linspace(0.04, 0.14, 26)

    for weights_prefix in itertools.product(weight_grid, repeat=n-1):
        w_sum = sum(weights_prefix)
        wn = 1.0 - w_sum

        # 2. Check the boundary condition for the last weight
        if wn < -1e-6:
            continue
        wn = max(0.0, wn)

        # 3. Create the final list of weights
        weights = list(weights_prefix) + [wn]

        for pct in pct_grid:
            total_error = 0
            for stroke in strokes:
                # Run CPD with current adaptive window percentage
                predicted_idx = find_change_point(S(stroke, weights), fraction, window_pct=pct)
                tail_cut = stroke.props["cuts"][1]
                total_error += abs(predicted_idx - tail_cut)

            mae = total_error / len(strokes)

            if mae < best_mae:
                best_mae = mae
                best_params = {
                    'weights': [round(w, 3) for w in weights],
                    # 'weights': [round(w, 3).item() for w in weights],
                    # 'weights': (round(w1, 2), round(w2, 2), round(w3, 2)),
                    'window_pct': round(pct, 3)
                }

    return best_params, best_mae
