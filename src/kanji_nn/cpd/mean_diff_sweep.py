import numpy as np

def mean_diff_sweep(S, k=8):
    n = len(S)
    scores = np.zeros(n)
    # Only sweep the last 40% of the stroke where pre-pen-up happens
    start_idx = int(n * 0.6)

    for t in range(start_idx, n - k):
        left_mean = np.mean(S[t-k:t])
        right_mean = np.mean(S[t:t+k])
        scores[t] = abs(left_mean - right_mean)

    return np.argmax(scores)
