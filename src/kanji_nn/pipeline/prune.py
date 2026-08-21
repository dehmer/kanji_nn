import numpy as np
from kanji_nn.data import Stroke


def prune(stroke, epsilon=1e-9):
    """
    Prune consecutive vertices with zero net displacement
    while keeping the first of a run.
    """
    t = stroke.features["t"]
    xy = stroke.features["xy"]
    pressure = stroke.features["pressure"]

    diffs = np.diff(xy, axis=0)
    mask = np.any(np.abs(diffs) > epsilon, axis=1)
    keep = np.concatenate(([True], mask))

    t = t[keep]
    xy = xy[keep]
    pressure = pressure[keep]

    return Stroke(
        dataset=stroke.dataset,
        key=stroke.key,
        sticky=stroke.sticky,
        features={"t": t, "xy": xy, "pressure": pressure}
    )
