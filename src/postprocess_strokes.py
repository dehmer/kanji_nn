#!/usr/bin/env python3

import os
import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from more_itertools import partition
from signal import signal, SIGINT

import kanji_nn.pipeline as pipeline
from kanji_nn.plot import strokes_plot, paths_plot
from kanji_nn.conditioning import join_strokes
from kanji_nn.data import Character, Stroke
from kanji_nn.data import plot_mcp
from kanji_nn.io.WKBReader import WKBReader
from kanji_nn.predef import compose, tap


plot_channels=["angle"]

def compose_pipeline(wkb_reader):
    return compose(
        # # tap(partial(plot_mcp, show=True, save=False, channels=plot_channels)),
        # pipeline.curve_fitting,
        # pipeline.resolve_segments,
        # pipeline.dtw,
        # pipeline.turning_angle,
        # pipeline.arc_length_raw,
        # partial(pipeline.gauss_1d, sigma=3.0),
        # # NOTE: after this point stroke lost all props/features.
        # partial(pipeline.reset, key="resampled:xy"),
        # pipeline.resampling_uniform,
        # pipeline.arc_length_raw,
        # # NOTE: after this point stroke lost all props/features.
        partial(pipeline.reset, key="trimmed:xy"),
        pipeline.trim_region,
        pipeline.vg_trace_align,
        partial(pipeline.wkb, wkb_reader=wkb_reader),
        pipeline.local_straightness,
        partial(pipeline.tangential_acc, speed_key="raw:speed:central"),
        pipeline.vector_acc,
        pipeline.curvature,
        pipeline.tangent,
        pipeline.central_speed,
        pipeline.straightness,
        pipeline.arc_length,
        pipeline.pressure_derivative,
        pipeline.pressure
    )


def process_file(dataset, pipeline, wkb_reader, filename):

    # reflect, nearest, mirror
    char = Character.of_npy(dataset, filename)
    strokes = char.strokes()

    trimmed = [pipeline(s) for s in strokes]
    # reference = wkb_reader[char.code_point][1]
    # paths = [s.sticky["path"] for s in trimmed]
    # fitted_paths = [s.props["fitted_path"] for s in trimmed]

    # struts = np.concatenate([s.props["struts"] for s in trimmed])
    # gauss_xy = [s.features["gauss:xy"] for s in trimmed]

    # styles = [
    #     {"color": "green", "linewidth": 1.0, "alpha": 1.0},
    #     {"color": "black", "linewidth": 1.0, "alpha": 1.0},
    # ]

    # paths_plot(fitted_paths, xdim=1.0, ydim=1.0, show_obb=False)
    # paths_plot(paths)
    # strokes_plot.overlays([reference, xy], styles, struts)
    # strokes_plot.overlays([xy, gauss_xy], styles, struts=struts)
    # strokes_plot.show(xy, alpha=0.0)
    # strokes_plot.quads(trimmed, figsize=(18, 18))

    trimmed_xy = [s.xy for s in trimmed]
    plot_filename = f'data/dataset/{dataset}/png-post/{char.code_point}-PP'
    strokes_plot.save(plot_filename, trimmed_xy, alpha=0.0)


def literal_to_hex(literal):
    return f'{ord(literal):x}'.upper()

def infer_file_names(literals):
    return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]

if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())

    white_list = []
    # white_list = infer_file_names("虫")
    # white_list = infer_file_names("ね")
    # white_list = infer_file_names("ま")

    datasets = ['katakana_47', 'hiragana_46', 'kanken-10_80']
    for dataset in datasets:
        npy_raw = f'data/dataset/{dataset}/npy-raw'
        wkb_reader = WKBReader(f"data/dataset/{dataset}/wkb", dataset)
        p = compose_pipeline(wkb_reader)

        for (dirpath, dirnames, filenames) in os.walk(npy_raw):
            for filename in filenames:
                if not filename.endswith('npy'): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, p, wkb_reader, f'{dirpath}/{filename}')
