import numpy as np
from svg.path import Path


def paths_to_array(paths: list[Path]) -> np.ndarray:
    """
    Convert path list with respective initial Move and
    consecutive CubicBezier segments into compact ndarray
    of shape (N, 8 + 1) with x/y for p0, p1, p2, p3 and
    additional pen-down (1)/pen-up (0) column.
    N is the total number of CubicBezier segments over all
    paths.
    """
    segments = []
    for path in paths:
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

    return np.vstack(segments)
