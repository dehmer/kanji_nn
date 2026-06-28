#!/usr/bin/env python3

import sys
from os import walk
import re
import numpy as np

INPUT_BASE_DIR = '/Users/dehmer/Public/Data/kanji_nn/strokes'

def extract_code_point(filename):
    match = re.search('.*(U\\+[0-9A-F]{4,5})', filename)
    if not match:
        raise Exception(f'cp: invalid format in {filename}')
    return match.group(1)

def cp_to_chr(cp):
    return chr(int(cp[2:], 16))

def split_strokes(raw):
    # drop pen status and split into strokes
    split_indices = np.where(raw[:, 4] == 0)[0] + 1
    raw = raw[:, :-1]
    return np.split(raw, split_indices[:-1])

def format_line(cp, literal, stroke_count, strokes):
    strokes = [np.array2string(stroke[:,[1,2]], separator=', ') for stroke in strokes]
    strokes = ', '.join(strokes).replace('\n', '')
    return f"{literal}\t{cp}\t{stroke_count}\t[{strokes}]"

def process(filename):

    # shape (n, 5) float32
    # columns:
    #   0: time_stamp
    #   1: x coord [0..1]
    #   2: y coord [0..1]
    #   3: pressure [0.4..~2.8]
    #   4: status feature (pen down/up) [0, 1]
    raw = np.load(filename)
    cp = extract_code_point(filename)
    literal = cp_to_chr(cp)
    strokes = split_strokes(raw)
    stroke_count = len(strokes)
    line = format_line(cp, literal, stroke_count, strokes)
    print(line)

if __name__ == "__main__":
    if (len(sys.argv)) < 2:
        raise Exception('no set name given')

    set_name = sys.argv[1]
    for (dirpath, dirnames, filenames) in walk(f"{INPUT_BASE_DIR}/{set_name}"):
        for filename in filenames:
            if not filename.endswith('.npy'): continue
            # if filename != 'U+307F.npy': continue
            process(f'{dirpath}/{filename}')
