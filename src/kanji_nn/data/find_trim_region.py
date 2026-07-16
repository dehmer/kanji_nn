import numpy as np


default_detector = lambda stroke: stroke.clone(props = {"cuts": (0, stroke.n_points)})


def CJK_STROKE_H(stroke):
    ds = stroke.features["ds"]
    # candidate_head = stroke.props["candidate_head"]
    # candidate_tail = stroke.props["candidate_tail"]

    ds_max_idx = np.argmax(ds)
    head_mask = ds[:ds_max_idx] > 0
    tail_mask = ds[ds_max_idx:] > 0
    head_cut = len(head_mask) - 1 - np.argmax(~head_mask[::-1]) # inclusive
    tail_cut = np.argmin(tail_mask) + ds_max_idx + 1 # exclusive
    return stroke.clone(props = {"cuts": (head_cut, tail_cut)})


detectors = {
    "CJK STROKE H": CJK_STROKE_H
}


def find_trim_region(stroke):
    _, _, name = stroke.stroke_type
    detector = detectors.get(name, default_detector)
    return detector(stroke)
