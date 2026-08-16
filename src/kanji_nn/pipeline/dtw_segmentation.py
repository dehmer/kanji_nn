import numpy as np
from dtw import dtw, asymmetric

def dtw_segmentation(stroke):
    query = stroke.xy
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
    D = np.column_stack([W.index1, W.index2, s[W.index2]]).astype(np.int32)

    # Extract start/end indices for segment runs:

    idx = np.flatnonzero(np.diff(D[:, 2]))
    start_idx = np.concatenate(([0], idx + 1))
    end_idx = np.concatenate((idx, [len(D) - 1])) # inclusive

    qs = D[start_idx, 0]
    qe = D[end_idx, 0] + 1 # exclusive
    rs = D[start_idx, 1]

    # Prepare struts columns (for plotting): x1, y1, x2, y2 of segment starts:
    struts = np.column_stack([
        query[qs, 0],
        reference[rs, 0],
        query[qs, 1],
        reference[rs, 1]
    ])

    # Split query vertices into segments:
    idxs = np.column_stack([qs, qe])
    segments = [query[qs[i]:qe[i], :] for i in range(len(qs))]
    seg_pts = [pts[:, 0] + 1j * pts[:, 1] for pts in segments]
    props = {"D": D, "struts": struts, "segments": seg_pts}
    return stroke.clone(props=props)
