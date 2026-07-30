import numpy as np
from dtw import dtw, asymmetric, symmetricP2
import matplotlib.pyplot as plt
from svg.path import Move
from scipy.signal import savgol_filter
import kanji_nn.svg as svg

def dtw_segmentation(stroke):
    query = stroke.features["savgol:hf:xy"]
    xys = stroke.props["path:xys"]
    reference = xys[:, :-1]

    W = dtw(
        query,
        reference,
        dist_method="euclidean",
        step_pattern=asymmetric,
        open_begin=False,
        open_end=False,
        keep_internals=False
    )

    # NOTE: Explicitly overwrite first segment index to 1 (from 0).
    # We don't want to deal with Move segment (of length 0).
    #
    s = xys[:, -1].copy() # better safe then sorry (mutated below)
    s[0] = 1
    alignment = np.column_stack([W.index1, W.index2, s[W.index2]])

    # Extract start/end indices for segment runs:
    idx = np.where(np.diff(alignment[:, 2]) != 0)[0]
    start_idx = np.concatenate(([0], idx + 1))
    end_idx = np.concatenate((idx, [len(alignment) - 1]))
    starts = alignment[start_idx, 0]
    ends = alignment[end_idx, 0] + 1

    segments = np.column_stack([starts, ends])
    print(segments)

    return stroke
