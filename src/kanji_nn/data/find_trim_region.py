import numpy as np


default_detector = lambda stroke: stroke.clone(props = {"cuts": (0, stroke.n_points)})


def CJK_STROKE_H(stroke):
    ds = stroke.features["ds"]
    ds_max_idx = np.argmax(ds)

    head_mask = ds[:ds_max_idx] > 0
    if np.all(head_mask):
        head_cut = 0
    else:
        head_cut = len(head_mask) - 1 - np.argmax(~head_mask[::-1])  # inclusive

    tail_mask = ds[ds_max_idx:] > 0
    if np.all(tail_mask):
        tail_cut = stroke.n_points
    else:
        tail_cut = np.argmin(tail_mask) + ds_max_idx + 1 # exclusive

    return stroke.clone(props = {"cuts": (head_cut, tail_cut)})


detectors = {
    "CJK STROKE H": CJK_STROKE_H
}


def find_trim_region(stroke):
    _, _, name = stroke.stroke_type
    detector = detectors.get(name, default_detector)
    return detector(stroke)
