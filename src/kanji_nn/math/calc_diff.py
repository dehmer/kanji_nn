import numpy as np

def calc_diff(blocks):
    # Calculate and prepend diffs with np.nan row (shift 1 to right).
    deltas = np.diff(blocks['raw'], axis=0)
    nans = np.full((1, deltas.shape[1]), np.nan, dtype=np.float32)
    padded = np.vstack((nans, deltas))
    return {'diff': padded}
