import re
import numpy as np
import unicodedata
import os

from ..conditioning import split_strokes, join_strokes
from .stroke import Stroke
from kanji_nn.io.KanjiVG import KanjiVG
from kanji_nn.svg.transform import transform


def extract_code_point(filename):
    match = re.search(r"(U\+[0-9A-F]{4,5})", filename)
    if not match:
        raise ValueError(f'code_point: invalid format in {filename}')
    return match.group(1)


class Character:
    def __init__(self, dataset, code_point, raw):
        """
        Fixed five column layout for raw (for now):
        0: timestamp (t)
        1: x coordinates
        2: y coordinates
        3: pressure
        4: pen-down/-up
        """
        self.dataset = dataset
        self.code_point = code_point
        self.raw = raw
        self.literal = chr(int(code_point[2:], 16))

    @classmethod
    def of_npy(cls, dataset, filename):
        code_point = extract_code_point(filename)
        raw = np.load(filename)

        # Silently drop orientation and tilt:
        if raw.shape[1] == 7:
            raw = raw[:, (0, 1, 2, 3, 6)]

        return cls(dataset, code_point, raw)

    @classmethod
    def of_strokes(cls, dataset, strokes):
        code_point = strokes[0].code_point
        strokes = [stroke.raw for stroke in strokes]
        raw = join_strokes(strokes)
        return cls(dataset, code_point, raw)

    def strokes(self):
        kvg = KanjiVG(self.code_point)
        strokes = split_strokes(self.raw)
        paths = [transform(path, a=1/109) for path in kvg.paths]

        return [Stroke(
            dataset=self.dataset,
            key=f"{self.literal}/{i}/{len(strokes)}",
            raw=raw,
            sticky={
                "kvg_type": kvg.types[i],
                "path": paths[i]
            }
        ) for i, raw in enumerate(strokes)]

    def stroke_types(self):
        """
        Lazily fetches kvg:type info for this character's strokes.
        Returns {stroke_idx: (type_literal, type_cp)}.
        """
        if getattr(self, "_stroke_types", None) is None:
            self._stroke_types = fetch_stroke_types(self.literal)
        return self._stroke_types