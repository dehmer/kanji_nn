import numpy as np

def dtw_align(sample_xy, ref_xy, band=None):
    """
    Classic DTW between two point sequences of arbitrary (possibly unequal)
    length, using Euclidean distance as the local cost.

    Returns:
        path: list of (i, j) index pairs, from (0,0) to (N-1, M-1),
              monotonic in both i and j (allows many-to-one both ways).
        cost: the cumulative cost matrix (N+1, M+1), 1-indexed internally.
    """
    N, M = len(sample_xy), len(ref_xy)
    diff = sample_xy[:, None, :] - ref_xy[None, :, :]
    local_cost = np.linalg.norm(diff, axis=2)  # (N, M)

    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, N + 1):
        j_range = range(1, M + 1)
        if band is not None:
            center = i * M / N
            j_range = [j for j in j_range if abs(j - center) <= band]
        for j in j_range:
            cost = local_cost[i - 1, j - 1]
            D[i, j] = cost + min(D[i - 1, j],      # sample pt unmatched (insertion)
                                  D[i, j - 1],      # ref pt unmatched (deletion)
                                  D[i - 1, j - 1])  # 1:1 (substitution)

    i, j = N, M
    path = []
    while (i, j) != (0, 0):
        path.append((i - 1, j - 1))
        candidates = [
            (D[i - 1, j - 1], (i - 1, j - 1)),
            (D[i - 1, j],     (i - 1, j)),
            (D[i, j - 1],     (i, j - 1)),
        ]
        _, (i, j) = min(candidates, key=lambda c: c[0])
    path.reverse()

    return path, D
