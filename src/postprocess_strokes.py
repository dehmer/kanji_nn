#!/usr/bin/env python3

from os import walk
import re
from functools import reduce
import numpy as np
from kanji_nn.plot import Kinematics
from kanji_nn.math import calc_kinematic
from kanji_nn.conditioning import *

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


def compose(*functions):
    """Composes functions from right to left."""
    return lambda x: reduce(lambda acc, f: f(acc), reversed(functions), x)


def split_strokes(raw):
    # drop pen status and split into strokes
    split_indices = np.where(raw[:, 4] == 0)[0] + 1
    raw = raw[:, :-1]
    return np.split(raw, split_indices[:-1])


def join_strokes(strokes):
    chunks = []
    for stroke in strokes:
        # pen status: 111....110
        status = np.ones((len(stroke), 1), dtype=np.float32)
        status[-1] = 0
        chunk = np.hstack((stroke, status))
        chunks.append(chunk)
    return np.vstack(chunks)


def process(dirpath, filename):
    code_point = extract_code_point(filename)
    literal = cp_to_chr(code_point)
    print(f'{code_point} {literal}')

    # 0: timestamp, 1: x, 2: y, 3: pressure, 4: feature
    raw = np.load(f'{dirpath}/{filename}')

    process_stroke = compose(
        # simplify_rdp,
        dehook,
        smooth_gaussian
    )

    process_character = compose(
        join_strokes,
        # lambda strokes: simplify_rdp(strokes, epsilon=rdp_epsilon),
        # lambda strokes: chaikin_smooth(strokes, refinements=chaikin_refinements),
        # lambda strokes: resample_kanji_spline(strokes, n_points=32),
        lambda strokes: [process_stroke(stroke) for stroke in strokes],
        split_strokes
    )

    raw = process_character(raw)

    configs = [
        dict(color="tab:gray", label="Pressure", data=lambda blocks: blocks['raw'][:, 3], linestyle="-."),
        dict(color="tab:blue", label="Velocity", data=lambda blocks: blocks['velocity']),
        dict(color="tab:red", label="Angle", data=lambda blocks: blocks['angle']),
    ]

    kinematics = Kinematics(raw, configs, figsize=(18, 12))
    kinematics.show()

    # blocks = {'raw': raw}
    # blocks = blocks | calc_kinematic(blocks)
    # interactive_kanji(blocks, configs, figsize=(18, 12))
    exit()

if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=240,  formatter=dict(float=lambda x: "%.3g" % x))
    np.set_printoptions(edgeitems=30, linewidth=240)
    DIR_NAME = 'data/katakana_49/strokes'

    white_list = []
    white_list = infer_file_names(['イ'])

    for (dirpath, dirnames, filenames) in walk(DIR_NAME):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if len(white_list) and not filename in white_list: continue
            process(dirpath, filename)
