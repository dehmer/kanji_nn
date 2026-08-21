import numpy as np
import kanji_nn.svg as svg

path_fn = lambda s: s.sticky["path"]

def resample_path_equidistant(stroke, path_fn=path_fn, factor=1.0, error=1e-5):
    path = path_fn(stroke)
    s = stroke.features["raw:s"]

    # increase/decrease density compared to stroke:
    ds = s[-1] / len(s)
    path_length = path.length(error=error)
    n_out = round(path_length / (ds * factor))

    xys = svg.resample_equidistant(path, n_out, error=error)
    return stroke.clone(props={"path:xys": xys}, force=True)
