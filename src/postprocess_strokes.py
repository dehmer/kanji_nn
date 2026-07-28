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
        data.curve_fitting,
        data.resolve_segments,
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
    fitted_paths = [s.props["fitted_path"] for s in trimmed]

    struts = np.concatenate([s.props["struts"] for s in trimmed])
    gauss_xy = [s.features["gauss:xy"] for s in trimmed]
    xy = [s.xy for s in trimmed]

    styles = [
        {"color": "green", "linewidth": 1.0, "alpha": 1.0},
        {"color": "black", "linewidth": 1.0, "alpha": 1.0},
    ]

    paths_plot(fitted_paths, xdim=1.0, ydim=1.0, show_obb=False)
    # paths_plot(paths)
    # strokes_plot.overlays([reference, xy], styles, struts)
    # strokes_plot.overlays([xy, gauss_xy], styles, struts=struts)
    # strokes_plot.show(xy, alpha=0.0)
    # strokes_plot.quads(trimmed, figsize=(18, 18))

    # plot_filename = f'data/dataset/{dataset}/png-post/{char.code_point}'
    # strokes_plot.save(plot_filename, trimmed, alpha=0.1)


def literal_to_hex(literal):
    return f'{ord(literal):x}'.upper()

def infer_file_names(literals):
    return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]

if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())

    white_list = []
    # white_list = infer_file_names("虫")
    white_list = infer_file_names("ね")
    # white_list = infer_file_names("ま")

    datasets = ['katakana_47', 'hiragana_46', 'kanken-10_80']
    for dataset in datasets:
        npy_raw = f'data/dataset/{dataset}/npy-raw'
        wkb_reader = WKBReader(f"data/dataset/{dataset}/wkb", dataset)
        pipeline = compose_pipeline(wkb_reader)

        for (dirpath, dirnames, filenames) in os.walk(npy_raw):
            for filename in filenames:
                if not filename.endswith('npy'): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, pipeline, wkb_reader, f'{dirpath}/{filename}')
