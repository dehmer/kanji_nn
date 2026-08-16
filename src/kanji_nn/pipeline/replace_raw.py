import numpy as np
from kanji_nn.data import Stroke

def replace_raw(stroke, raw_fn):
    return Stroke(
        dataset=stroke.dataset,
        key=stroke.key,
        raw=raw_fn(stroke),
        sticky=stroke.sticky
    )
