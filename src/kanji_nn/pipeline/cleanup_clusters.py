import numpy as np
import math
from functools import partial
import scipy.signal as signal
from kanji_nn.predef import clamp
from kanji_nn.data import Stroke


def arc_length_fractions(points):
    diffs = np.diff(points, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    return cum / total if total > 0 else np.linspace(0, 1, len(points))


def repair_leg_pair(xy, i_in, peak, i_out, apex):
    """
    Rebuild i_in->apex->i_out as two straight legs. i_in/i_out stay
    untouched; interior samples are repositioned at their original
    fractional arc-length progress, projected onto the new geometry.
    """
    p_in, p_out = xy[i_in], xy[i_out]
    entry_frac = arc_length_fractions(xy[i_in:peak + 1])
    exit_frac = arc_length_fractions(xy[peak:i_out + 1])
    print("entry_frac", entry_frac)
    print("exit_frac", exit_frac)

    new_xy = xy.copy()
    print("in [old]:", i_in + 1, "->", peak + 1, "\n", new_xy[i_in + 1:peak + 1])
    new_xy[i_in + 1:peak + 1] = p_in + entry_frac[1:, None] * (apex - p_in)
    print("in [new]:", i_in + 1, "->", peak + 1, "\n", new_xy[i_in + 1:peak + 1])

    print("out [old]", peak, "->", i_out, "\n", new_xy[peak:i_out])
    new_xy[peak:i_out] = apex + exit_frac[:-1, None] * (p_out - apex)
    print("out [new]", peak, "->", i_out, "\n", new_xy[peak:i_out])
    return new_xy


def cleanup_clusters(stroke):
    """
    Cleanup clusters according their classification.
    This step ultimately creates a new stroke with updated raw data.
    """

    clusters = stroke.props.get("clusters", [])
    if not clusters:
        return stroke

    # ア/0-shaped case only: single cluster, single peak. Anything else
    # is deferred until classification exists.
    (i_in, i_out), c = next(iter(clusters.items()))
    if len(c["peaks"]) != 1 or c["designated_apex"] is None:
        return stroke

    xy = repair_leg_pair(stroke.xy, i_in, c["peaks"][0], i_out, c["designated_apex"])
    raw = stroke.raw.copy()
    raw[:, 1:3] = xy

    return Stroke(
        dataset=stroke.dataset,
        key=stroke.key,
        raw=raw,
        sticky=stroke.sticky
    )