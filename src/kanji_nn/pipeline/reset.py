import numpy as np
from kanji_nn.data import Stroke

def reset(stroke, key):
    """
    Create new stroke from selected xy property.
    Timestamp is set to [0, n_points).
    Properties and features are cleared out, only stickies are retained.
    """

    xy = stroke.props[key]
    raw = np.column_stack([
        np.arange(xy.shape[0]), # fake timestamp
        xy,
        np.zeros(xy.shape[0]) # pressure
    ])

    return Stroke(
        dataset=stroke.dataset,
        stroke_index=stroke.stroke_index,
        raw=raw,
        code_point=stroke.code_point,
        literal=stroke.literal,
        sticky=stroke.sticky
    )
