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

    ds = stroke.features["raw:ds"]
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

    weights = [0.95, 0.05] # optimized
    window_pct = 0.04
    fraction = (1.0 - 0.15)

    S = stroke.props["S"]
    tail_cut = find_change_point(S(weights), fraction, window_pct)

    return stroke.clone(props = {"cuts": (head_cut, int(tail_cut))}, force=True)

def CJK_STROKE_GENERIC(stroke):
    call_site = "CJK_STROKE_GENERIC"

    # HEAD CUT =>

    ds = get_channel(stroke, "raw:ds", call_site)
    ds_max_idx = np.argmax(ds)
    mask = (ds > 0)
    find_index = lambda rng: next((i for i in rng if not mask[i]), rng[-1])

    ranges = [
        range(ds_max_idx - 1, -1, -1),     # backward search, left slope.
        range(ds_max_idx, stroke.n_points) # forward search, right slope.
    ]

    cuts = tuple(find_index(r) for r in ranges)
    cuts = (cuts[0], cuts[1] + 1) # [h, t] -> [h, t)
    head_cut = cuts[0]

    # TAIL CUT =>

    weights = [0.95, 0.05] # optimized
    window_pct = 0.04
    fraction = (1.0 - 0.15)
    S = stroke.props["S"]
    tail_cut = int(find_change_point(S(weights), fraction, window_pct))

    return stroke.clone(props = {"cuts": (head_cut, tail_cut)})


detectors = {
    # "H": CJK_STROKE_H,
    # "S": CJK_STROKE_H,
    # "P": CJK_STROKE_H,
    # "D": CJK_STROKE_H,
    # "N": CJK_STROKE_H,
    # "T": CJK_STROKE_H,
    # "HZ": CJK_STROKE_HZ,
    # "HG": CJK_STROKE_HZ,
    # "SG": CJK_STROKE_HZ,
    # "SW": CJK_STROKE_HZ,
    # "WG": CJK_STROKE_HZ,
    # "HP": CJK_STROKE_HZ,
    # "PZ": CJK_STROKE_HZ,
    # "PD": CJK_STROKE_HZ,
    # "SP": CJK_STROKE_HZ,
}

"""
issues:
U+5B50, U+5B57, U+5B66, U+5C0F, U+5C71, U+6C34,
U+7AF9, U+7CF8, U+8D64, U+51FA, U+529B, U+6C17,
U+753A, U+5915
"""

def find_trim_region(stroke):
    _, _, name = stroke.stroke_type
    detector = detectors.get(name, CJK_STROKE_GENERIC)
    # detector = detectors.get(name, default_detector)
    return detector(stroke)
