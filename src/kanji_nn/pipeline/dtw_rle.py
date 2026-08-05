import numpy as np
from dtw import dtw, asymmetric
from itertools import accumulate
from scipy.signal import find_peaks


def signature(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.empty((0, 4), dtype=int)

    # Locate where consecutive elements change value
    changes = mask[:-1] != mask[1:]

    # Get indices of transitions and pad with boundaries
    idx = np.flatnonzero(changes) + 1
    splits = np.concatenate(([0], idx, [mask.size]))

    # Compute run properties
    starts = splits[:-1]
    ends = splits[1:]
    lengths = ends - starts
    values = mask[starts].astype(int)

    # Stack into a 2D ndarray
    return np.column_stack((values, starts, ends, lengths))


def reduce(s, i, j):
    gap = 1
    if i == j: return i, j
    elif s[i][0]: return reduce(s, i + 1, j)
    elif s[j][0]: return reduce(s, i, j - 1)
    elif s[i][3] <= gap: return reduce(s, i + 1, j)
    elif s[j][3] <= gap: return reduce(s, i, j - 1)
    else: return (i, j)


def dtw_rle(stroke):
    """
    Wording:
        * query, reference, (optomized) warping path
        * one-to-many "stagnation" points
    """

    query = stroke.xy
    path_xys = stroke.props["path:xys"]
    reference = path_xys[:, :-1]

    W = dtw(
        query,
        reference,
        dist_method="euclidean",
        step_pattern=asymmetric,  # Crucial for open-ended alignment
        open_begin=True,          # Relaxes the starting boundary condition
        open_end=True,            # Relaxes the ending boundary condition
        keep_internals=False
    )

    # Create "stroke signature" with respective run lengths for T/F groups:
    mask = np.diff(W.index2) == 0
    s = signature(mask)

    i, j = 0, len(s) - 1

    # unconditional. cut dirty at 0 and n-1
    if s[i][0]: i = i + 1
    if s[j][0]: j = j - 1

    i, j = reduce(s, i, j)

    head_cut = 0 if i == 0 else int(s[i-1][2]) # inclusive
    tail_cut = int(s[j][2]) + 1                # exclusive
    cuts = (head_cut, tail_cut)

    return stroke.clone(props={"cuts": cuts})
