import numpy as np
import kanji_nn.svg as svg

def resample_path_equidistant(stroke, factor=1.0, error=1e-5):
    path = stroke.sticky["path"]
    s = stroke.features["raw:s"]

    # increase density compared to stroke:
    ds = s[-1] / stroke.n_points
    path_length = path.length(error=error)
    n_out = round(path_length / (ds * factor))

    xys = svg.resample_equidistant(path, n_out, error=error)
    return stroke.clone(props={"path:xys": xys})
