import numpy as np
from kanji_nn.data import Stroke

def replace_xy(stroke, key):
    """
    Create new stroke from selected xy property.
    Keep timestamp/pressure as-is.
    Properties and features are cleared out, only stickies are retained.
    """

    xy = stroke.features[key]
    raw = np.column_stack([
        stroke.t,
        xy,
        stroke.pressure
    ])

    return Stroke(
        dataset=stroke.dataset,
        key=stroke.key,
        raw=raw,
        sticky=stroke.sticky
    )
