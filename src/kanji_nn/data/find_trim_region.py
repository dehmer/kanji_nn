import numpy as np


default_detector = lambda stroke: stroke.clone(props = {"cuts": (0, stroke.n_points)})


def CJK_STROKE_H(stroke):
    """
    Find head/tail cut locations for simple shaped
    strokes without 跳ね (hane) and 折れ (ore).
    Tested to work well for CJK STROKE H, S, P, and D.

    Adds "cuts" tuple as result to stroke.
    """

    # don't overwrite presets.
    if "cuts" in stroke.props:
        return stroke

    ds = stroke.features["ds"]
    assert len(ds) != 0, "[CJK_STROKE_H] degenerate stroke detected (empty ds)"

    ds_max_idx = np.argmax(ds)
    assert ds_max_idx != 0, "[CJK_STROKE_H] unexpected condition (ds_max_idx == 0)"

    # forward/backward search first False, return range end if not found.
    mask = (ds > 0)
    find_index = lambda rng: next((i for i in rng if not mask[i]), rng[-1])

    ranges = [
        range(ds_max_idx - 1, -1, -1),     # backward search, left slope.
        range(ds_max_idx, stroke.n_points) # forward search, right slope.
    ]

    cuts = tuple(find_index(r) for r in ranges)
    cuts = (cuts[0], cuts[1] + 1) # [h, t] -> [h, t)
    return stroke.clone(props = {"cuts": cuts})

detectors = {
    "CJK STROKE H": CJK_STROKE_H,
    "CJK STROKE S": CJK_STROKE_H,
    "CJK STROKE P": CJK_STROKE_H,
    "CJK STROKE D": CJK_STROKE_H,

    # CJK_STROKE_H doesn't work well for:
    # (P_norm is still high in ds troves)
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
