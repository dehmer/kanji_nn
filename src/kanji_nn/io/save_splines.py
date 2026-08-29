import numpy as np
from kanji_nn.bezier.spline_array import paths_to_array

path_fn = lambda s: s.sticky["path"]


def save_splines(filename_fn, path_fn=path_fn):
    """
    Save CubicBezier segments as flat 2D ndarray:
    8 columns for control points x/y + pen-down/-up column
    """
    stroke_count = 0
    paths = []
    def inner(stroke):
        nonlocal paths
        paths.append(path_fn(stroke))

        if len(paths) == stroke.stroke_count:
            filename = filename_fn(stroke)
            raw = paths_to_array(paths)
            np.save(filename, raw)
            paths = []

        return stroke
    return inner
