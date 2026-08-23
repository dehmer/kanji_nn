import numpy as np
from kanji_nn.conditioning import join_strokes


path_fn = lambda s: s.sticky["path"]


def save_splines(dirname, path_fn=path_fn):
    """
    Save CubicBezier segments as flat 2D ndarray:
    8 columns for control points x/y + pen-down/-up column
    """
    paths = []
    def inner(stroke):
        nonlocal paths
        path = path_fn(stroke)

        segments = []
        # skip Move segment
        for i in range(1, len(path)):
            s = path[i] # bezier segment
            pen = 0 if i == len(path) - 1 else 1

            segments.append(np.array([
                s.start.real,    s.start.imag,
                s.control1.real, s.control1.imag,
                s.control2.real, s.control2.imag,
                s.end.real,      s.end.imag,
                pen
            ]))

        paths.append(np.vstack(segments))

        if len(paths) == stroke.stroke_count:
            filename = f"data/dataset/{stroke.dataset}/{dirname}/{stroke.code_point}.npy"

            raw = np.vstack(paths)
            print(raw.shape)
            np.save(filename, raw)
            paths = []
        return stroke
    return inner
