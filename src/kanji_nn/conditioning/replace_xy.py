import numpy as np
from kanji_nn.data import Stroke


def replace_xy(stroke, key):
    """
    Create new stroke from selected xy property.
    Keep timestamp/pressure as-is.
    Properties and features are cleared out, only stickies are retained.
    """

    xy = stroke.features[key]
    t = np.arange(0, len(xy))
    pressure = np.zeros(len(xy))

    return Stroke(
        dataset=stroke.dataset,
        key=stroke.key,
        sticky=stroke.sticky,
        features={"t": t, "xy": xy, "pressure": pressure}
    )
