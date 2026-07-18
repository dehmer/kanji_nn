import numpy as np

# 2. Adaptive Mean-Difference Sweep CPD Engine
def find_change_point_adaptive(S, window_pct=0.08, min_w=4, max_w=12):
    """
    Finds the change point using a window size that scales with stroke length.

    window_pct: Percentage of total stroke length to use for window size.
    min_w: Hard minimum window size (prevents noise sensitivity on short strokes).
    max_w: Hard maximum window size (prevents signal bleeding on long strokes).
    """
    n = len(S)
    scores = np.zeros(n)

    # Calculate adaptive window size based on individual stroke length
    window_size = int(round(n * window_pct))
    window_size = max(min_w, min(window_size, max_w))

    # Focus exclusively on the last 35% of the stroke to find the pre-pen-up zone
    start_search_idx = int(n * 0.65)

    for t in range(start_search_idx, n - window_size):
        left_window = S[t - window_size : t]
        right_window = S[t : t + window_size]

        # Maximize the difference between the two adaptive windows
        scores[t] = abs(np.mean(left_window) - np.mean(right_window))

    return np.argmax(scores)
