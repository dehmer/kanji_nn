import numpy as np

# 2. Mean-Difference Sweep CPD Engine
def find_change_point(S, window_size=6):
    """
    Finds the index that maximizes the change in mean between two adjacent windows.
    Focuses exclusively on the last 40% of the stroke to avoid writing pauses.
    """
    n = len(S)
    scores = np.zeros(n)
    start_search_idx = int(n * 0.6)

    for t in range(start_search_idx, n - window_size):
        left_window = S[t - window_size : t]
        right_window = S[t : t + window_size]

        # Maximize absolute difference in local means
        scores[t] = abs(np.mean(left_window) - np.mean(right_window))

    return np.argmax(scores)
