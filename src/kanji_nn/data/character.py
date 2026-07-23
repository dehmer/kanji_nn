import re
import numpy as np
import psycopg
import unicodedata
import os

from .identity import identity
from ..conditioning import split_strokes, join_strokes
from .stroke import Stroke
from .conninfo_from_env import conninfo_from_env

_connection = None

def get_connection():
    global _connection
    if _connection is None:
        _connection = psycopg.connect(conninfo_from_env())
    return _connection

def stroke_type_name(type_cp):
    """type_cp like 'U+31D6' -> 'HG'"""

    # Kana without stroke types:
    if not type_cp:
        return 'N/A'

    ch = chr(int(type_cp[2:], 16))
    name = unicodedata.name(ch)
    return name.split(' ')[2]

def fetch_stroke_types(literal):
    """
    Returns {stroke_idx: (type_literal, type_cp)} for the given character
    literal. Empty dict if nothing is found (e.g. hiragana/katakana, which
    KanjiVG does not annotate with stroke types).
    """
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT stroke_idx, type_literal, code_point "
            "FROM kvg_type WHERE literal = %s",
            (literal,)
        )
        return {row[0]: (row[1], row[2], stroke_type_name(row[2])) for row in cur.fetchall()}


def extract_code_point(filename):
    match = re.search(r"(U\+[0-9A-F]{4,5})", filename)
    if not match:
        raise ValueError(f'cp: invalid format in {filename}')
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

        # Silently drop orientation and tile:
        if raw.shape[1] == 7:
            raw = raw[:, (0, 1, 2, 3, 6)]

        return cls(dataset, code_point, raw)

    @classmethod
    def of_strokes(cls, dataset, strokes):
        code_point = strokes[0].code_point
        strokes = [stroke.raw for stroke in strokes]
        raw = join_strokes(strokes)
        return cls(dataset, code_point, raw)

    # Stick either to raw xy or smoothened xy (not both).
    def strokes(self, smooth_fn = identity):
        strokes = split_strokes(self.raw)
        types = self.stroke_types()
        assert len(strokes) == len(types), "stroke/type count mismatch"

        def smooth(raw):
            xy = smooth_fn(raw[:, 1:3])
            return np.column_stack((raw[:, 0], xy, raw[:, 3:]))

        return [Stroke(
            dataset = self.dataset,
            stroke_index = i,
            raw = smooth(raw),
            code_point = self.code_point,
            literal = self.literal,
            stroke_type = types[i]
        ) for i, raw in enumerate(strokes)]

    def stroke_types(self):
        """
        Lazily fetches kvg:type info for this character's strokes.
        Returns {stroke_idx: (type_literal, type_cp)}.
        """
        if getattr(self, "_stroke_types", None) is None:
            self._stroke_types = fetch_stroke_types(self.literal)
        return self._stroke_types