import numpy as np
from kanji_nn.conditioning import join_strokes


raw_fn = lambda s: s.features["xy"]

def save_npy(dirname, raw_fn=raw_fn):
    strokes = []
    def inner(stroke):
        nonlocal strokes
        strokes.append(stroke)

        if len(strokes) == stroke.stroke_count:
            filename = f"data/dataset/{stroke.dataset}/{dirname}/{stroke.code_point}.npy"
            raw = [raw_fn(s) for s in strokes]
            raw = join_strokes(raw)
            np.save(filename, raw)
            strokes = []
        return stroke
    return inner
