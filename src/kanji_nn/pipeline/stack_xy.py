import numpy as np
from kanji_nn.data import Stroke


def stack_xy(stroke, key):
    xy = stroke.features[key]
    raw = np.column_stack([stroke.t, xy, stroke.pressure])

    return Stroke(
        dataset=stroke.dataset,
        stroke_index=stroke.stroke_index,
        raw=raw,
        code_point=stroke.code_point,
        literal=stroke.literal,
        sticky=stroke.sticky
    )
