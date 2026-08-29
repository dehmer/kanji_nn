import numpy as np


def paths_to_array(paths):
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

    segments = np.vstack(segments)
    return segments
