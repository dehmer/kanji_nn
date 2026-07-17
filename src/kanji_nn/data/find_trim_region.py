from dataclasses import dataclass
import numpy as np


default_detector = lambda stroke: stroke.clone(props = {"cuts": (0, stroke.n_points)})


@dataclass(frozen=True)
class Params:
    mask: object
    sign: int
    offset: int
    default: int


def _cut(params):
    if np.all(params.mask):
        return params.default
    else:
        return int(params.sign * np.argmin(params.mask) + params.offset)


def CJK_STROKE_H(stroke):
    ds = stroke.features["ds"]
    ds_max_idx = np.argmax(ds)

    mask = (ds > 0)
    head_mask = mask[:ds_max_idx]
    tail_mask = mask[ds_max_idx:]

    params = [
        Params(head_mask[::-1], -1, ds_max_idx - 1, 0),
        Params(tail_mask[::1], 1, ds_max_idx + 1, stroke.n_points)
    ]

    cuts = tuple(_cut(p) for p in params)
    return stroke.clone(props = {"cuts": cuts})


detectors = {
    "CJK STROKE H": CJK_STROKE_H,
    "CJK STROKE S": CJK_STROKE_H,
    "CJK STROKE P": CJK_STROKE_H,
    "CJK STROKE D": CJK_STROKE_H,

    # dont't work well:
    # (P_norm still up in ds trove)
    "CJK STROKE HG": CJK_STROKE_H,
    "CJK STROKE WG": CJK_STROKE_H,
    "CJK STROKE N": CJK_STROKE_H,
    "CJK STROKE HZ": CJK_STROKE_H,
    "CJK STROKE SG": CJK_STROKE_H,
    "CJK STROKE SWG": CJK_STROKE_H,
}


def find_trim_region(stroke):
    _, _, name = stroke.stroke_type
    detector = detectors.get(name, default_detector)
    return detector(stroke)
