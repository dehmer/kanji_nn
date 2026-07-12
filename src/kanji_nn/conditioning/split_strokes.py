import numpy as np

def split_strokes(raw):
    """
    Drop pen status and split into strokes.
    Assumes pen state as last column.
    """
    split_indices = np.where(raw[:, -1] == 0)[0] + 1
    raw = raw[:, :-1]
    return np.split(raw, split_indices[:-1])
