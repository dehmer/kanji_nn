#!/usr/bin/env python3

import os
import re
from functools import reduce, partial
import numpy as np
from kanji_nn.plot import Kinematics, literal, plot_stroke_geometry
import kanji_nn.math as math
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

def cut_hooks(stroke):
    geometry = math.geometry(stroke)
    start, end = detect_hooks(geometry)
    # plot_stroke_geometry([geometry])
    return stroke[start:end]

process_stroke = compose(
    partial(simplify_rdp, epsilon=0.005),
    partial(smooth_chaikin, refinements=1),
    cut_hooks,
    # lambda strokes: strokes[:, [1, 2]],
    partial(simplify_rdp, xy_cols=(1, 2), epsilon=0.0005),
    partial(smooth_gaussian, xy_cols=(1, 2)),
)

process_character = compose(
    join_strokes,
    lambda strokes: [process_stroke(stroke) for stroke in strokes],
    split_strokes
)

def process(output_dir, dirpath, filename):
    code_point = extract_code_point(filename)
    literal = cp_to_chr(code_point)
    print(f'{code_point} {literal}')

    # 0: timestamp, 1: x, 2: y, 3: pressure, 4: feature
    raw = np.load(f'{dirpath}/{filename}')
    # character.show(raw)
    strokes = process_character(raw)
    # character.show(strokes)
    literal.save(f'data/{output_dir}/png-post/{code_point}', strokes)

if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=240,  formatter=dict(float=lambda x: "%.3g" % x))
    np.set_printoptions(edgeitems=30, linewidth=240)
    input_dir = 'data/npy.(5)-raw/kanken-10_80'
    output_dir = 'kanken-10_80'

    dirs = [
        f'data/{output_dir}',
        f'data/{output_dir}/npy.(5)-post',
        f'data/{output_dir}/png-post'
    ]

    for dir in dirs:
        if os.path.exists(dir): continue
        os.mkdir(dir)

    white_list = []
    # white_list = infer_file_names(['目'])

    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if len(white_list) and not filename in white_list: continue
            process(output_dir, dirpath, filename)