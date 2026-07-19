import numpy as np

# 2. Mean-Difference Sweep CPD Engine
def find_change_point(S, fraction=(1 - 0.15), window_pct=0.08, min_w=4, max_w=10):
    """
    Finds the index that maximizes the change in mean between two adjacent windows.
    """
    n = len(S)
    scores = np.zeros(n)

    # Calculate adaptive window size based on individual stroke length
    window_size = int(round(n * window_pct))
    window_size = max(min_w, min(window_size, max_w))
    start_search_idx = int(n * fraction)

    for t in range(start_search_idx, n - window_size):
        left_window = S[t - window_size : t]
        right_window = S[t : t + window_size]

        # Maximize absolute difference in local means
        scores[t] = abs(np.mean(left_window) - np.mean(right_window))

    return np.argmax(scores)
