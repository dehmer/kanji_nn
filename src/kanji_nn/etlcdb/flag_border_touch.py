import numpy as np


def flag_border_touch(glyph):
    data = np.array(glyph["image:binary"].convert("L"))

    border = np.concatenate([
        data[0, :], data[-1, :], data[:, 0], data[:, -1]
    ])

    if np.any(border > 0):
        return glyph | {"skip": True}

    return glyph