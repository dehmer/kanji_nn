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


def find_tail_trough(c_speed, ds_max_idx):
    """
    Finds the LAST local minimum (trough) in the tail segment: the last
    point where c_speed stops declining and turns to rise again. This is
    different from the global minimum, which can lock onto an earlier,
    lower-valued pause (e.g. a mid-stroke corner) and miss the real,
    later decay that precedes the genuine tail.

    Returns len(c_speed) if no local minimum is found before the
    recording ends (speed still falling, or still rising throughout ->
    no cut needed).
    """
    tail = c_speed[ds_max_idx:]
    n = len(tail)

    last_trough_idx = None
    falling = False  # have we seen a decline since the last trough?

    for i in range(1, n):
        if tail[i] < tail[i - 1]:
            falling = True
        elif tail[i] > tail[i - 1] and falling:
            # turned upward after a decline -> i-1 was a local minimum
            last_trough_idx = i - 1
            falling = False

    if last_trough_idx is None:
        return len(c_speed)  # no real trough found -> no cut

    return ds_max_idx + last_trough_idx + 1  # exclusive tail_cut

def get_channel(stroke, key, call_site="unknown"):
    channel = stroke.features[key]
    assert len(channel) == stroke.n_points, f"[{call_site}] channel sample count mismatch: {key}"
    return channel


def CJK_STROKE_HG(stroke):
    head_cut = 0
    if "cuts" in stroke.props:
        head_cut, _ = stroke.props["cuts"]

    call_site = "CJK_STROKE_HG"
    ds = get_channel(stroke, "ds", call_site)
    c_speed = get_channel(stroke, "c_speed", call_site)

    ds_max_idx = np.argmax(ds)
    assert ds_max_idx != 0, f"[{call_site}] unexpected condition (ds_max_idx == 0)"

    tail_cut = find_tail_trough(c_speed, ds_max_idx)

    return stroke.clone(props = {"cuts": (head_cut, int(tail_cut))}, force=True)

detectors = {
    "CJK STROKE H": CJK_STROKE_H,
    "CJK STROKE S": CJK_STROKE_H,
    "CJK STROKE P": CJK_STROKE_H,
    "CJK STROKE D": CJK_STROKE_H,

    # test CJK_STROKE_HG on remaining stroke type as well
    "CJK STROKE HG": CJK_STROKE_HG,
    "CJK STROKE WG": CJK_STROKE_HG,
    "CJK STROKE N": CJK_STROKE_HG,
    "CJK STROKE HZ": CJK_STROKE_HG,
    "CJK STROKE SG": CJK_STROKE_HG,
    "CJK STROKE SWG": CJK_STROKE_HG,
}


def find_trim_region(stroke):
    _, _, name = stroke.stroke_type
    detector = detectors.get(name, default_detector)
    return detector(stroke)
