#!/usr/bin/env python3

import os
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from more_itertools import partition

from kanji_nn.plot import multi_channel_plot, strokes_plot
from kanji_nn.conditioning import join_strokes
from kanji_nn.data import compose, identity, Character, Stroke
import kanji_nn.metrics as metrics


threshold=0.2
composed_metrics = compose(
    metrics.consolidate_edges,
    partial(metrics.detect_edges, signal_key='loc_stness', threshold=threshold),
    partial(metrics.detect_edges, signal_key='stness', threshold=threshold),
    metrics.loc_stness,
    metrics.pressure_derivative,
    partial(metrics.tangential_acc, speed_key="c_speed"),
    metrics.vector_acc,
    metrics.curvature,
    metrics.tangent,
    metrics.central_speed,
    metrics.straightness,
    metrics.arc_length
)

def process_file(dataset, filename):
    SIGMA = 1.0
    smooth_fn = lambda xy: gaussian_filter1d(xy, SIGMA, axis=0, mode='nearest') # reflect, nearest, mirror
    char = Character.of_npy(filename)

    strokes = char.strokes(smooth_fn=identity)
    # strokes = char.strokes(smooth_fn=smooth_fn)

    trimmed_strokes = []
    for stroke in strokes:

        # add pressure as explicit feature for plot:
        stroke = stroke.clone(features={"P": stroke.pressure})
        trimmed_stroke = composed_metrics(stroke)
        trimmed_strokes.append(trimmed_stroke)

        # channels = ["P", "stness", "c_speed", "θ", "dθ/ds", "K",]
        # channels = ["P", "dP/dt", "stness", "loc_stness", "c_speed", "at", "axy"]
        channels = ["P", "dP/dt", "stness", "loc_stness", "c_speed"]

        if not trimmed_stroke.isbare:
            multi_channel_plot(trimmed_stroke, channels, figsize=(18, 8))
            plt.show()

    trimmed_strokes = [stroke.raw[:, 1:] for stroke in trimmed_strokes]
    # print(trimmed_strokes)
    # exit()
    filename = f'data/dataset/{dataset}/png-post/{char.code_point}'
    # strokes_plot.show(trimmed_strokes)
    strokes_plot.save(filename, trimmed_strokes)

if __name__ == "__main__":
    dataset = 'katakana_49'
    dataset = 'hiragana_48'
    dataset = 'kanken-10_80'
    in_dir = f'data/dataset/{dataset}/npy-raw'

    def literal_to_hex(literal):
        return f'{ord(literal):x}'.upper()

    def infer_file_names(literals):
        return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]

    white_list = []
    white_list = infer_file_names('字')
    # white_list = infer_file_names('字学左文村校森玉空貝赤足音')

    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if len(white_list) and not filename in white_list: continue
            process_file(dataset, f'{dirpath}/{filename}')
