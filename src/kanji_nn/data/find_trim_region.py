from dataclasses import dataclass
import numpy as np


default_detector = lambda stroke: stroke.clone(props = {"cuts": (0, stroke.n_points)})


def CJK_STROKE_H(stroke):
    ds = stroke.features["ds"]
    assert len(ds) != 0, "CJK_STROKE_H requires a non-degenerate stroke (empty ds)"

    ds_max_idx = np.argmax(ds)
    mask = (ds > 0)

    def find_index(rng):
        for i in rng:
            if not mask[i]:
                break
        return i

    ranges = [
        range(ds_max_idx - 1, -1, -1),
        range(ds_max_idx, stroke.n_points)
    ]

    cuts = tuple(find_index(r) for r in ranges)
    cuts = (cuts[0], cuts[1] + 1)
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
