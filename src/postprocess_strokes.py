#!/usr/bin/env python3

import os
import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from more_itertools import partition
from signal import signal, SIGINT

from kanji_nn.plot import strokes_plot, paths_plot
from kanji_nn.conditioning import join_strokes
from kanji_nn.data import compose, identity, tap, Character, Stroke
from kanji_nn.data import trim_region, plot_mcp
import kanji_nn.metrics as metrics
from kanji_nn.io.WKBReader import WKBReader
import kanji_nn.data as data


plot_channels=["angle"]


def compose_pipeline(wkb_reader):
    return compose(
        # tap(partial(plot_mcp, show=True, save=False, channels=plot_channels)),
        data.dtw,
        data.turning_angle,
        metrics.arc_length_raw,
        partial(data.gauss_1d, sigma=3.0),
        data.resampling_uniform,
        metrics.arc_length_raw,
        # NOTE: after this point stroke lost all props/features.
        trim_region,
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


def process_file(dataset, pipeline, wkb_reader, filename):

    # reflect, nearest, mirror
    char = Character.of_npy(dataset, filename)
    strokes = char.strokes()

    trimmed = [pipeline(s) for s in strokes]
    reference = wkb_reader[char.code_point][1]

    paths = [s.sticky["path"] for s in trimmed]
    paths_plot(paths)

    struts = np.concatenate([s.props["struts"] for s in trimmed])
    trimmed = [s.features["gauss:xy"] for s in trimmed]

    styles = [
        {"color": "green", "linewidth": 1.0, "alpha": 1.0},
        {"color": "black", "linewidth": 1.0, "alpha": 1.0},
    ]
    strokes_plot.overlays([reference, trimmed], styles, struts)

    plot_filename = f'data/dataset/{dataset}/png-post/{char.code_point}'
    strokes_plot.save(plot_filename, trimmed, alpha=0.1)


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())
    # dataset = 'katakana_47'
    dataset = 'hiragana_46'
    # dataset = 'kanken-10_80'
    in_dir = f'data/dataset/{dataset}/npy-raw'

    wkb_reader = WKBReader(f"data/dataset/{dataset}/wkb", dataset)
    pipeline = compose_pipeline(wkb_reader)

    def literal_to_hex(literal):
        return f'{ord(literal):x}'.upper()

    def infer_file_names(literals):
        return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]

    white_list = []
    # white_list = infer_file_names("虫")
    # white_list = infer_file_names("ね")
    white_list = infer_file_names("ま")

    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            # if len(white_list) and not filename in white_list: continue
            if white_list and filename not in white_list: continue
            process_file(dataset, pipeline, wkb_reader, f'{dirpath}/{filename}')
