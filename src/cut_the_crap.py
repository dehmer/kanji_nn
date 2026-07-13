#!/usr/bin/env python3

import os
from functools import partial
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import csv

from kanji_nn.plot import multi_channel_plot, strokes_plot
from kanji_nn.conditioning import split_strokes
from kanji_nn.data import compose, identity, Character, Stroke
import kanji_nn.metrics as metrics


def process_file(filename):

    metrics_composed = compose(
        metrics.curvature,
        metrics.tangent,
        metrics.central_speed,
        metrics.straightness,
        metrics.arc_length
    )

    char = Character.of_npy(filename)
    strokes = char.strokes(smooth_fn=identity)
    for stroke in strokes:

        # add pressure as explicit feature for plot:
        stroke = stroke.clone(features={"pressure": stroke.pressure})
        stroke = metrics_composed(stroke)

        channels = ["pressure", "straightness", "speed", "θ", "dθ/ds", "K",]
        multi_channel_plot(stroke, channels, figsize=(18, 8))
        # plt.savefig("kinematics")
        plt.show()


if __name__ == "__main__":
    SIGMA = 1.0
    smooth_fn = lambda xy: gaussian_filter1d(xy, SIGMA, axis=0, mode='nearest') # reflect, nearest, mirror

    rows = []
    with open('data/analysis-short-samples.csv', mode='r', newline='', encoding='utf-8') as file:
        dict_reader = csv.DictReader(file)

        columns = ['literal', 'code_point', 'stroke_idx', 'duration', 'head_cut', 'tail_cut']
        for row in dict_reader:
            rows.append(row)

    filename = lambda cp: f'data/dataset/kanken-10_80/npy-raw/{cp}.npy'
    code_points = set([row['code_point'] for row in rows])
    characters = [Character.of_npy(filename(cp)) for cp in code_points]

    # plot and save raw data:
    for character in characters:
        strokes = character.strokes()
        filename = f"data/dataset/short-strokes/png-raw/{character.code_point}"
        strokes_plot.save(filename, strokes, 'raw')

    # mutable (sic)
    strokes_lookup = { c.code_point: c.strokes() for c in characters}

    # trim strokes:
    for row in rows:
        cp, idx = row['code_point'], int(row['stroke_idx'])

        # hc, tc: 0-based timestamp offsets
        hc = int(row['head_cut']) if row['head_cut'] else None
        tc = int(row['tail_cut']) if row['tail_cut'] else None
        strokes_lookup[cp][idx] = strokes_lookup[cp][idx].trim(hc, tc)

    for cp in code_points:
        strokes = strokes_lookup[cp]
        filename = f"data/dataset/short-strokes/png-post/{cp}"
        strokes_plot.save(filename, strokes, 'raw/trimmed')
        # character = Character.of_strokes(strokes_lookup)
        # strokes_lookup = character.strokes(smooth_fn)
    exit()


    dataset = 'katakana_49'
    dataset = 'hiragana_48'
    dataset = 'kanken-10_80'
    in_dir = f'data/dataset/{dataset}/npy-raw'

    def literal_to_hex(literal):
        return f'{ord(literal):x}'.upper()

    def infer_file_names(literals):
        return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]

    white_list = []
    white_list = infer_file_names(['音'])

    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if len(white_list) and not filename in white_list: continue
            process_file(f'{dirpath}/{filename}')
