#!/usr/bin/env python3

import os
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from more_itertools import partition
from signal import signal, SIGINT

from kanji_nn.plot import strokes_plot
from kanji_nn.conditioning import join_strokes
from kanji_nn.data import compose, identity, tap, Character, Stroke
from kanji_nn.data import find_trim_region, trim_region, plot_mcp
import kanji_nn.metrics as metrics
from kanji_nn.io.WKBReader import WKBReader
import kanji_nn.data as data


wkb_reader = WKBReader('data/dataset/kanken_6355/wkb', 'kanken_6355')
plot_channels=["P", "raw:ds", "raw:speed:central", "gauss:θ", "dirty"]

composed_metrics = compose(
    # NOTE: after this point stroke lost all props/features.
    trim_region,
    # tap(partial(plot_mcp, show=True, save=False, channels=plot_channels)),
    data.vg_trace_align,
    partial(data.wkb, wkb_reader=wkb_reader),
    metrics.local_straightness,
    partial(metrics.tangential_acc, speed_key="raw:speed:central"),
    metrics.vector_acc,
    metrics.curvature,
    metrics.tangent,
    metrics.central_speed,
    metrics.straightness,
    metrics.arc_length,
    metrics.pressure_derivative,
    metrics.pressure
)


def process_file(dataset, filename):

    # reflect, nearest, mirror
    char = Character.of_npy(dataset, filename)
    strokes = char.strokes(smooth_fn=identity)

    strokes = [composed_metrics(s) for s in strokes]
    strokes = [s.raw[:, 1:] for s in strokes]
    filename = f'data/dataset/{dataset}/png-post/{char.code_point}'
    # strokes_plot.show(trimmed_strokes, alpha=0.1)
    strokes_plot.save(filename, strokes, alpha=0.1)


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: exit())
    # dataset = 'katakana_49'
    # dataset = 'hiragana_48'
    dataset = 'kanken-10_80'
    in_dir = f'data/dataset/{dataset}/npy-raw'

    def literal_to_hex(literal):
        return f'{ord(literal):x}'.upper()

    def infer_file_names(literals):
        return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]

    white_list = []
    # white_list = infer_file_names("字学左文村校森玉空貝赤足音") # shorties
    # white_list = infer_file_names("人休入八大天文木本林校森森水火犬耳虫足金") # CJK STROKES N, T
    # white_list = infer_file_names("中五口右名四日早田男町白百目石草虫見貝足車音") # CJK STROKE HZ
    # white_list = infer_file_names("九円")

    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if len(white_list) and not filename in white_list: continue
            process_file(dataset, f'{dirpath}/{filename}')
