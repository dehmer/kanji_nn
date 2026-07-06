import numpy as np

def split_strokes(raw):
    # drop pen status and split into strokes
    split_indices = np.where(raw[:, 2] == 0)[0] + 1
    raw = raw[:, :-1]
    return np.split(raw, split_indices[:-1])
