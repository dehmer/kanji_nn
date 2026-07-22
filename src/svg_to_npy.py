#!/usr/bin/env python3

import csv
import numpy as np
import shapely.wkb
from shapely.geometry import LineString, MultiLineString
from svg.path import Move, CubicBezier
import itertools
from pstats import SortKey, Stats

from kanji_nn.io import parse_archive, write_wkb
from kanji_nn.conditioning import sample_kanjivg_char, join_strokes
from kanji_nn.io import code_point
import kanji_nn.io.filters as filters

def flatten(xs):
    return list(itertools.chain.from_iterable(xs))


def mk_cp(i): return f'U+{i:x}'.upper()


joyo_lookup = { chr(o):True for o in code_point.lookups['JOYO']}


kanken_lookup = {
    chr(o): True for o in flatten([
        code_point.lookups[key] for key in [
            'KANKEN_10',
            'KANKEN_09',
            'KANKEN_08',
            'KANKEN_07',
            'KANKEN_06',
            'KANKEN_05',
            'KANKEN_04',
            'KANKEN_03',
            'KANKEN_PRE_2',
            'KANKEN_02',
            'KANKEN_PRE_1',
            'KANKEN_01',
        ]
    ])
}


joyo_filter = lambda literal, _ : literal in joyo_lookup
kanken_filter = lambda literal, _ : literal in kanken_lookup


def literal_filter(xs):
    lookup = { literal:True for literal in xs}
    return lambda literal, _ : literal in lookup


def all_literals(literal, groups):
    return True


if __name__ == '__main__':
    ZIP_PATH = "/Users/dehmer/Public/Data/kanjivg-20250816-all.zip"
    dataset = "kanken_6355"
    output_dir = f"data/dataset/{dataset}/npy-vg"

    for index, entry in enumerate(parse_archive(ZIP_PATH, filter=filters.kanji_kanken)):
        label = entry[0]
        paths = entry[1]
        strokes = sample_kanjivg_char(paths, tol=0.05)
        strokes = join_strokes(strokes)
        print(index, label, len(strokes))
        np.save(f'{output_dir}/{label}-vg.npy', strokes)
