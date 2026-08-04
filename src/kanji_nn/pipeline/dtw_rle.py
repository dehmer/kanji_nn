import numpy as np
from dtw import dtw, asymmetric
from itertools import accumulate
from scipy.signal import find_peaks


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


    # Create "stroke signature" with respective run lengths
    # and cumulative run length sums for T/F groups:
    mask = np.diff(W.index2) == 0
    edges = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1, len(mask)]
    s = list(zip(mask[edges[:-1]], np.diff(edges)))

    s = [
        (tag, rl, W.index1[cs]) for (tag, rl), cs in zip(
            s,
            accumulate(length for _, length in s)
        )
    ]

    i, j = 0, len(s) - 1 # [i, j]

    # unconditional. cut dirty at 0 and n-1
    if s[i][0]: i = i + 1
    if s[j][0]: j = j - 1

    # jump small clean gaps (length < max_gap)
    max_gap = 2
    if s[i][1] < max_gap: i = i + 2
    if s[j][1] < max_gap: j = j - 2

    head_cut = 0 if i == 0 else int(s[i-1][2]) + 1 # inclusive
    tail_cut = int(s[j][2]) + 1                    # exclusive

    # To prevent overcut (inside a small window) snap
    # to nearest turn angle peak in outward direction.

    angle = stroke.features["angle:w=1:abs"]
    window = 8
    peaks, _ = find_peaks(angle, prominence=0.09)

    head_diff = head_cut - peaks
    head_indices = np.flatnonzero((head_diff > 0) & (head_diff <= window))
    if len(head_indices):
        head_cut = int(peaks[head_indices[-1]] + 1)

    tail_diff = peaks - tail_cut
    tail_indices = np.flatnonzero((tail_diff > 0) & (tail_diff <= window))
    if len(tail_indices):
        tail_cut = int(peaks[tail_indices[0]])

    cuts = (head_cut, tail_cut)

    return stroke.clone(props={"cuts": cuts})
