#!/usr/bin/env python3

import csv
import numpy as np
import shapely.wkb
from shapely.geometry import LineString, MultiLineString
from kanji_nn.io import parse_archive, write_wkb
import kanji_nn.io.filters as filters

def all_literals(literal, groups):
    return True


def split_strokes(raw):
    # drop pen status and split into strokes
    split_indices = np.where(raw[:, 2] == 0)[0] + 1
    raw = raw[:, :-1]
    return np.split(raw, split_indices[:-1])


if __name__ == '__main__':
    ZIP_PATH = "/Users/dehmer/Public/Data/kanjivg-20250816-all.zip"
    OUTPUT_DIR = "data/wkb"
    MAX_POINTS = 80 # kanji
    # MAX_POINTS = 32 # hiragana

    filter = filters.kanji_kanken
    for label, paths in parse_archive(ZIP_PATH, filter):
        print(label)

    # zipped = zip(labels, strokes)
    # kanji_data = [{"label": item[0], "strokes": split_strokes(item[1])} for item in zipped]
    # write_wkb(kanji_data, OUTPUT_DIR, "katakana_47")
