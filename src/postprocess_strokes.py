#!/usr/bin/env python3

import os
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from more_itertools import partition

from kanji_nn.plot import multi_channel_plot, strokes_plot
from kanji_nn.conditioning import split_strokes
from kanji_nn.data import compose, identity, Character, Stroke
import kanji_nn.metrics as metrics

def combined_straightness(stroke):
    # 1. Initialize output signal with all 1.0
    a = stroke.features['straightness']
    b = stroke.features['local_straightness']
    c = np.ones_like(a)

    # 2. Create masks for where either signal hits 0.0 or 1.0
    mask_min = (a == 0.0) | (b == 0.0)
    mask_max = (a == 1.0) | (b == 1.0)

    # 3. Apply the values directly to the output signal
    c[mask_min] = 0.0
    c[mask_max] = 1.0

    return stroke.clone(features={'combined_straightness': c})


threshold=0.2
composed_metrics = compose(
    metrics.consolidate_edges,
    partial(metrics.detect_edges, signal_key='local_straightness', threshold=threshold),
    partial(metrics.detect_edges, signal_key='straightness', threshold=threshold),
    metrics.local_straightness,
    metrics.pressure_derivative,
    partial(metrics.tangential_acc, speed_key='central_speed'),
    metrics.vector_acc,
    metrics.curvature,
    metrics.tangent,
    metrics.central_speed,
    metrics.straightness,
    metrics.arc_length
)

def process_file(filename):
    SIGMA = 1.0
    smooth_fn = lambda xy: gaussian_filter1d(xy, SIGMA, axis=0, mode='nearest') # reflect, nearest, mirror
    char = Character.of_npy(filename)

    strokes = char.strokes(smooth_fn=identity)
    # strokes = char.strokes(smooth_fn=smooth_fn)
    for stroke in strokes:

        # add pressure as explicit feature for plot:
        stroke = stroke.clone(features={"pressure": stroke.pressure})
        stroke = composed_metrics(stroke)

        # channels = ["pressure", "straightness", "central_speed", "θ", "dθ/ds", "K",]
        # channels = ["pressure", "dP/dt", "straightness", "local_straightness", "central_speed", "at", "axy"]
        channels = ["pressure", "dP/dt", "straightness", "local_straightness", "central_speed"]
        multi_channel_plot(stroke, channels, figsize=(18, 8))
        # plt.savefig("kinematics")
        plt.show()


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
    # white_list = infer_file_names('学')
    white_list = infer_file_names('字学左文村校森玉空貝赤足音')

    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if len(white_list) and not filename in white_list: continue
            process_file(f'{dirpath}/{filename}')
