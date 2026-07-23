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


plot_channels=["P", "raw:ds", "raw:speed:central", "gauss:θ"]


def compose_pipeline(wkb_reader):
    return compose(
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


def process_file(dataset, pipeline, filename):

    # reflect, nearest, mirror
    char = Character.of_npy(dataset, filename)
    strokes = char.strokes(smooth_fn=identity)

    reference = wkb_reader[char.code_point][1]
    trimmed = [pipeline(s) for s in strokes]
    trimmed = [s.raw[:, 1:] for s in trimmed]
    strokes = [s.raw[:, 1:] for s in strokes]
    # strokes_plot.show(reference, alpha=0.0)
    # strokes_plot.show(trimmed, alpha=0.0)
    # styles = [
    #     {"color": "royalblue", "linewidth": 5.0, "alpha": 0.5},
    #     {"color": "red", "linewidth": 2.0, "alpha": 1.0},
    #     {"color": "black", "linewidth": 2.0, "alpha": 1.0},
    # ]
    # strokes_plot.overlays([reference, strokes, trimmed], styles)
    # strokes_plot.show(trimmed, alpha=0.2)

    filename = f'data/dataset/{dataset}/png-post/{char.code_point}'
    strokes_plot.save(filename, trimmed, alpha=0.1)


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: exit())
    # dataset = 'katakana_47'
    # dataset = 'hiragana_46'
    dataset = 'kanken-10_80'
    in_dir = f'data/dataset/{dataset}/npy-raw'

    wkb_reader = WKBReader(f"data/dataset/{dataset}/wkb", dataset)
    pipeline = compose_pipeline(wkb_reader)

    def literal_to_hex(literal):
        return f'{ord(literal):x}'.upper()

    def infer_file_names(literals):
        return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]

    white_list = []
    # white_list = infer_file_names("字") # 学字校森林

    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if len(white_list) and not filename in white_list: continue
            process_file(dataset, pipeline, f'{dirpath}/{filename}')
