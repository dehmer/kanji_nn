import numpy as np
from kanji_nn.data import Stroke


def prune(stroke, epsilon=1e-9):
    """
    Prune consecutive vertices with zero net displacement
    while keeping the first of a run.
    """
    diffs = np.diff(stroke.xy, axis=0)
    mask = np.any(np.abs(diffs) > epsilon, axis=1)
    keep = np.concatenate(([True], mask))

    # Remove timestamp differences instead of absolute
    # timestamps than add up again.
    # Prepend original start time to differences:
    dt = np.concatenate(([stroke.t[0]], np.diff(stroke.t)))

    raw = np.column_stack([
        np.cumsum(dt[keep]),
        stroke.xy[keep],
        stroke.pressure[keep]
    ])

    return Stroke(
        dataset=stroke.dataset,
        key=stroke.key,
        raw=raw,
        sticky=stroke.sticky
    )
