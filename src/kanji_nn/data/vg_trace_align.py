import numpy as np
from kanji_nn.plot import strokes_plot
from . import dtw_align, densify

def cluster_mask(path, min_run=5):
    """
    [0.0, 1.0] channel, one value per stroke sample point, marking points
    that fall inside a DTW many-to-one cluster: a run of >= min_length
    consecutive path steps that all share the same reference index.
    """
    reference_idx = path[:, 1]

    changed = np.diff(reference_idx) != 0
    boundaries = np.where(np.concatenate(([True], changed, [True])))[0]
    run_lengths = np.diff(boundaries)

    # derive target length (len(stroke.xy)) from last stroke index in path:
    target_length = path[-1, 0] + 1
    mask = np.zeros(target_length, dtype=float)
    for b in np.where(run_lengths >= min_run)[0]:
        start, end = boundaries[b], boundaries[b + 1] - 1
        i0, i1 = path[start, 0], path[end, 0]
        mask[i0:i1 + 1] = 1.0
    return mask


# def fold_gaps(mask, gap_length=10):
#     """
#     Fold an initial gap into the head cluster: if the first True/1.0 run in
#     `mask` starts at or before `gap_length`, overwrite mask[0:gap_length]
#     with 1.0 -- absorbing the short "clean" prefix that can precede a real
#     head cluster into the cluster itself.

#     Only the leftmost run needs checking: if it doesn't start within
#     gap_length, no later run possibly could either (they only sit further
#     from index 0).
#     """
#     ones = np.where(mask > 0)[0]
#     if len(ones) == 0:
#         return mask

#     print('ones', ones)

#     mask = mask.copy()
#     first_start, last_end = ones[0], ones[-1]

#     if first_start <= gap_length:
#         mask[0:gap_length] = 1.0

#     print('last_end', last_end)
#     print('len(mask)', len(mask))

#     if len(mask) - last_end <= gap_length:
#         mask[last_end:] = 1.0

#     return mask

def fold_gaps(mask, gap_length=10):
    # 1. Identify where 1s are located
    ones_indices = np.where(mask == 1)[0]

    if ones_indices.size > 0:
        mask = mask.copy()
        # 2. Calculate the length of leading and trailing gaps
        leading_gap = ones_indices[0]
        trailing_gap = len(mask) - 1 - ones_indices[-1]

        # 3. Fill gaps with 1 if they fall within the gap_length
        if leading_gap <= gap_length:
            mask[:leading_gap] = 1
        if trailing_gap <= gap_length:
            mask[len(mask) - trailing_gap:] = 1

    return mask


def cuts(mask, gap_length):
    # 1. Get the indices where 1s are located
    ones_indices = np.where(mask == 1)[0]

    if ones_indices.size == 0:
        return (None, None)

    first_1 = ones_indices[0]
    last_1 = ones_indices[-1]

    # Identify where the clusters break (differences greater than 1)
    diffs = np.diff(ones_indices)
    break_points = np.where(diffs > 1)[0]

    # 2. Determine the last index of the first cluster
    head = None
    if first_1 <= gap_length:
        if break_points.size > 0:
            head = int(ones_indices[break_points[0]])
        else:
            head = int(last_1)  # Only one single cluster exists in the mask

    # 3. Determine the first index of the last cluster
    tail = None
    trailing_gap = len(mask) - 1 - last_1
    if trailing_gap <= gap_length:
        if break_points.size > 0:
            tail = int(ones_indices[break_points[-1] + 1])
        else:
            tail = int(first_1)  # Only one single cluster exists in the mask

    return (head, tail)


def vg_trace_align(stroke):
    print("[vg_trace_align]", f"{stroke.literal}/{stroke.stroke_index}")


    # WARNING: construction site ahead =>

    # Add interpolated points to reference
    # to make up for sparse point distribution.
    # max_ds := target delta arc length:
    s = stroke.features["raw:s"]
    max_ds = s[-1] / stroke.n_points
    reference = stroke.props["wkb"]
    densified = densify(reference, max_ds)

    # [start/end) indices (into path) of clusters
    # with same reference index and certain run length n:
    path, _ = dtw_align(stroke.xy, densified)
    mask = cluster_mask(path)

    gap_length = 12
    mask = fold_gaps(mask, gap_length)
    head_cut, tail_cut = cuts(mask, gap_length)

    if not head_cut:
        head_cut = 0

    if tail_cut:
        tail_cut = tail_cut + 1
    else:
        tail_cut = stroke.n_points


    print(head_cut, tail_cut)
    return stroke.clone(features={"dirty": mask.astype(np.float32)}, props={"cuts": (head_cut, tail_cut)})

"""
Observations so far suggest that gap lengths are somewhat shorter than cluster lengths. But even if not, I'd like to have two knobs to adjust individually. As for the tail gap: Same/mirrored handling can possibly be applied, but I want to address head situation first. While it may be easy to add extending head cluster in cluster_mask, I propose a separate function fold_gaps taking the mask from cluster_mask and gap_length.  If the first cluster's (or actually any cluster's) start falls within gap_length , the mask range [0,gap_length] is overwritten with True values. If this does make sense, please execute and show me the code for fold_gaps.
"""
