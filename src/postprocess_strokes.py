#!/usr/bin/env python3

from os import walk
import re
import numpy as np
from kanji_nn.plot import plot_interactive_kanji


def extract_code_point(filename):
    match = re.search('.*(U\\+[0-9A-F]{4,5})', filename)
    if not match:
        raise Exception(f'cp: invalid format in {filename}')
    return match.group(1)


def cp_to_chr(cp):
    return chr(int(cp[2:], 16))


def literal_to_hex(literal):
    return f'{ord(literal):x}'.upper()


def infer_file_names(literals):
    return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]


def calc_deltas_and_distance(raw):
    """
    Parameters:
    raw (np.ndarray with shape (*, 5)):
    """

    # Calculate and prepend diffs with np.nan row (shift 1 to right).
    diffs = np.diff(raw, axis=0)
    nans = np.full((1, diffs.shape[1]), np.nan, dtype=np.float32)
    diffs = np.vstack((nans, diffs))

    # First point indices for strokes (excluding stroke 0):
    firsts = np.where(raw[:, 4] == 0)[0][:-1] + 1

    dt, dx, dy = diffs[:, 0], diffs[:, 1], diffs[:, 2]
    headings = np.arctan2(dy, dx)
    distances = np.sqrt(dx**2 + dy**2)
    distances[firsts] = 0.0


    return np.vstack((dt, dx, dy, distances, headings)).T


def process(dirpath, filename):
    code_point = extract_code_point(filename)
    literal = cp_to_chr(code_point)
    print(f'{code_point} {literal}')


    # 0: timestamp, 1: x, 2: y, 3: pressure, 4: feature
    raw = np.load(f'{dirpath}/{filename}')
    # 5: timestamp delta (dt), 6: dx, 7: dy, 8: distances, 9: headings
    deltas = calc_deltas_and_distance(raw)
    raw = np.hstack((raw, deltas))
    plot_interactive_kanji(raw)

    print(raw.shape)
    exit()



if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=240,  formatter=dict(float=lambda x: "%.3g" % x))
    np.set_printoptions(edgeitems=30, linewidth=240)
    DIR_NAME = 'data/katakana_49/strokes'

    white_list = []
    white_list = infer_file_names(['ナ'])
    white_list = infer_file_names(['ネ'])

    for (dirpath, dirnames, filenames) in walk(DIR_NAME):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if len(white_list) and not filename in white_list: continue

            process(dirpath, filename)
