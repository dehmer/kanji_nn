import re
import numpy as np
from .identity import identity
from ..conditioning import split_strokes
from .stroke import Stroke


def extract_code_point(filename):
    match = re.search(r"(U\+[0-9A-F]{4,5})", filename)
    if not match:
        raise ValueError(f'cp: invalid format in {filename}')
    return match.group(1)


class Character:
    def __init__(self, code_point, raw):
        """
        Fixed five column layout for raw (for now):
        0: timestamp (t)
        1: x coordinates
        2: y coordinates
        3: pressure
        4: pen-down/-up
        """
        self.code_point = code_point
        self.raw = raw
        self.literal = chr(int(code_point[2:], 16))

    @classmethod
    def of_npy(cls, filename):
        code_point = extract_code_point(filename)
        raw = np.load(filename)

        # Silently drop orientation and tile:
        if raw.shape[1] == 7:
            raw = raw[:, (0, 1, 2, 3, 6)]

        return cls(code_point, raw)

    # Stick either to raw xy or smoothened xy (not both).
    def strokes(self, smooth_fn = identity):
        strokes = split_strokes(self.raw)

        def smooth(raw):
            xy = smooth_fn(raw[:, 1:3])
            return np.column_stack((raw[:, 0], xy, raw[:, 3:]))

        return [Stroke(i, smooth(raw), self.code_point, self.literal) for i, raw in enumerate(strokes)]
