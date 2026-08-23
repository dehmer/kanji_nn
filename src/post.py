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
import kanji_nn.io as io
import kanji_nn.plot as plot
import kanji_nn.conditioning as conditioning
import kanji_nn.bezier as bezier
import kanji_nn.metrics as metrics


strokes_png    = lambda s, x: f"data/dataset/{s.dataset}/png-post/{s.code_point}-{x}"
png_raw        = lambda s: strokes_png(s, "A")
png_gauss      = lambda s: strokes_png(s, "B")
png_trimmed    = lambda s: strokes_png(s, "C")
png_simplified = lambda s: strokes_png(s, "D")
png_fitted     = lambda s: strokes_png(s, "E")


fitted_path = lambda s: s.props["fitted"]
xys = lambda s: s.props["path:xys"][:, :-1]


def compose_pipeline():
    ds = 0.006      # slightly below minimum of all strokes
    alpha = 0.1
    sigma = 2.0     # Gauss 1D Filter
    epsilon = 3e-4  # RDP
    maxError = 5e-4 # Schneider's Algorithm

    return compose(
        partial(plot.save_strokes_plot(filename_fn=png_fitted, xy_fn=xys, title=f"fitted @ {maxError=}", alpha=alpha)),
        partial(bezier.resample_fixed_distance, path_fn=fitted_path, ds=ds),
        partial(bezier.schneider, maxError=maxError),

        partial(plot.save_strokes_plot(filename_fn=png_simplified, title=f"simplified @ {epsilon=}", alpha=alpha)),
        partial(conditioning.simplify_rdp, epsilon=epsilon),

        partial(plot.save_strokes_plot(filename_fn=png_trimmed, title="trimmed", alpha=alpha)),
        conditioning.trim_region,
        conditioning.dtw_rle,
        partial(metrics.turning_angle, w=1),
        partial(bezier.resample_fixed_distance, ds=ds),

        # Note: gauss_1d works best for uniformly sampled point w.r.t to arc-length spacing.
        # Hence resample_xy_equidistant first with ds=0.006.
        partial(plot.save_strokes_plot(filename_fn=png_gauss, title=f"gauss @ {sigma=}", alpha=alpha)),
        partial(conditioning.replace_xy, key="gauss:xy"),
        partial(conditioning.gauss_1d, sigma=sigma, f=None),
        partial(conditioning.resample_xy_equidistant, ds=ds),

        conditioning.prune,
        partial(plot.save_strokes_plot(filename_fn=png_raw, title="raw", alpha=alpha)),
        conditioning.split_raw,
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

    white_list = []
    # white_list = infer_file_names("子")

    pipeline = compose_pipeline()
    for dataset in datasets:
        directory = f"data/dataset/{dataset}/npy-raw"

        for (dirpath, dirnames, filenames) in os.walk(directory):
            filenames.sort()
            for filename in filenames:
                if not filename.endswith("npy"): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, pipeline, f"{dirpath}/{filename}")
