#!/usr/bin/env python3
import os
import sys
from signal import signal, SIGINT
import numpy as np
from functools import partial
import csv
import math

from kanji_nn.data import Character
from kanji_nn.predef import compose, tap
import kanji_nn.pipeline as pipeline
import kanji_nn.io as io
import kanji_nn.plot as plot
import kanji_nn.conditioning as conditioning
import kanji_nn.svg as svg


plot_channels=[
    "angle:w=1"
]

strokes_png    = lambda s, x: f"data/dataset/{s.dataset}/png-post/{s.code_point}-{x}"
png_raw        = lambda s: strokes_png(s, "A")
png_gauss      = lambda s: strokes_png(s, "B")
png_trimmed    = lambda s: strokes_png(s, "C")
png_simplified = lambda s: strokes_png(s, "D")
png_fitted     = lambda s: strokes_png(s, "E")


plot_channels=[
    "kappa",
]


fitted_path = lambda s: s.props["fitted"]
xys = lambda s: s.props["path:xys"][:, :-1]


def compose_pipeline(cuts_target):
    alpha = 0.3
    sigma = 2.0     # Gauss 1D Filter
    epsilon = 3e-4  # RDP
    maxError = 5e-4 # Schneider's Algorithm

    return compose(
        partial(plot.save_strokes_plot(filename_fn=png_fitted, xy_fn=xys, title=f"fitted @ {maxError=}", alpha=alpha)),
        # plot.show_strokes_plot(xy_fn=xys, alpha=0.1),
        partial(pipeline.resample_path_equidistant, path_fn=fitted_path, factor=0.5),
        partial(svg.schneider, maxError=maxError),
        pipeline.arc_length_raw,

        partial(plot.save_strokes_plot(filename_fn=png_simplified, title=f"simplified @ {epsilon=}", alpha=alpha)),
        partial(conditioning.simplify_rdp, epsilon=epsilon),
        pipeline.arc_length_raw,

        partial(plot.save_strokes_plot(filename_fn=png_trimmed, title="trimmed", alpha=alpha)),
        pipeline.trim_region,
        pipeline.dtw_rle,
        partial(pipeline.turning_angle, w=1),
        pipeline.resample_path_equidistant,
        pipeline.arc_length_raw,

        # Note: gauss_1d works best for uniformly sampled point w.r.t to arc-length spacing.
        # Hence resample_xy_equidistant first with ds=0.006.
        partial(plot.save_strokes_plot(filename_fn=png_gauss, title=f"gauss @ {sigma=}", alpha=alpha)),
        partial(pipeline.replace_xy, key="gauss:xy"),
        partial(pipeline.gauss_1d, sigma=sigma, f=None),
        conditioning.resample_xy_equidistant,

        pipeline.prune,
        partial(plot.save_strokes_plot(filename_fn=png_raw, title="raw", alpha=alpha)),
        tap(lambda s: print(f"{s.dataset} - {s.literal}/{s.stroke_index}"))
    )


def process_file(dataset, pipeline, filename):
    character = Character.of_npy(dataset, filename)
    strokes = character.strokes()
    strokes = [pipeline(s) for s in strokes]


def literal_to_hex(literal):
    return f"{ord(literal):x}".upper()


def infer_file_names(literals):
    return [f"U+{literal_to_hex(literal)}.npy" for literal in literals]


def load_cuts(filename):
    cuts = {}
    with open(filename, mode="r", newline="", encoding="utf-8") as file:
        dict_reader = csv.DictReader(file)
        for row in dict_reader:
            key = row['stroke']
            value = (int(row["head"]), int(row["tail"]))
            cuts[key] = value

    return cuts


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())

    datasets = [
        "katakana_47",
        "hiragana_46",
        "kanken-10_80",
    ]

    cuts_target = load_cuts("data/cuts-baseline.csv")

    white_list = []
    # white_list = infer_file_names("子")

    for dataset in datasets:
        directory = f"data/dataset/{dataset}/npy-raw"
        p = compose_pipeline(cuts_target)

        for (dirpath, dirnames, filenames) in os.walk(directory):
            filenames.sort()
            for filename in filenames:
                if not filename.endswith("npy"): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, p, f"{dirpath}/{filename}")
