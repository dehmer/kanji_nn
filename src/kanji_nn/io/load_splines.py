import numpy as np
from svg.path import Path, Move, CubicBezier


def load_splines(filename):
    raw = np.load(filename)

    split_indices = np.where(raw[:, -1] == 0)[0] + 1
    raw = raw[:, :-1]
    path_list = np.split(raw, split_indices[:-1])

    paths = []
    for entry in path_list:
        segments = [Move(entry[0][0] + 1j * entry[0][1])]
        for cubic in entry:
            segments.append(CubicBezier(
                cubic[0] + 1j * cubic[1],
                cubic[2] + 1j * cubic[3],
                cubic[4] + 1j * cubic[5],
                cubic[6] + 1j * cubic[7],
            ))
        paths.append(Path(*segments))
    return paths
