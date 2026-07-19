import numpy as np
from kanji_nn.cpd import find_change_point

default_detector = lambda stroke: stroke.clone(props = {"cuts": (0, stroke.n_points)}, force=True)


def get_channel(stroke, key, call_site="unknown"):
    channel = stroke.features[key]
    assert len(channel) == stroke.n_points, f"[{call_site}] channel sample count mismatch: {key}"
    return channel


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


def CJK_STROKE_HZ(stroke):
    head_cut, tail_cut = 0, stroke.n_points
    if "cuts" in stroke.props:
        head_cut, tail_cut = stroke.props["cuts"]

    call_site = "CJK_STROKE_HZ"
    S = get_channel(stroke, "S", call_site)

    # construction site ahead =>
    fraction = (1.0 - 0.15)
    window_pct = 0.06
    # window_pct = 0.048
    tail_cut = find_change_point(S, fraction, window_pct)

    return stroke.clone(props = {"cuts": (head_cut, int(tail_cut))}, force=True)

detectors = {
    "H": CJK_STROKE_H,
    "S": CJK_STROKE_H,
    "P": CJK_STROKE_H,
    "D": CJK_STROKE_H,
    "N": CJK_STROKE_H,
    "T": CJK_STROKE_H,
    "HZ": CJK_STROKE_HZ,
}


def find_trim_region(stroke):
    _, _, name = stroke.stroke_type
    detector = detectors.get(name, default_detector)
    return detector(stroke)
