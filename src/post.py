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


plot_channels=[
    "angle:w=1"
]

sigma = 1.4
png_raw = lambda s: f"data/dataset/{s.dataset}/png-post/{s.code_point}-raw"
png_gauss = lambda s: f"data/dataset/{s.dataset}/png-post/{s.code_point}-gauss"
png_trimmed = lambda s: f"data/dataset/{s.dataset}/png-post/{s.code_point}-trimmed"


plot_channels=[
    "angle:w=1",
    "angle:w=5",
    "raw:speed:central",
    "tortuosity",
    "backtrack_fraction",
]


fitted_path = lambda s: s.props["fitted"]


def compose_pipeline(cuts_target):
    return compose(
        # plot.show_strokes_plot(lambda s: s.xy),
        # plot.save_strokes_plot("png-trimmed"),
        # io.save_npy("npy-trimmed"),
        # partial(plot.save_strokes_plot(filename_fn=png_trimmed, title="trimmed")),
        # plot.show_strokes_plot(),
        plot.show_paths_plot(path_fn=fitted_path, show_badges=False),
        pipeline.joint_refinement,
        pipeline.fit_segments,
        # pipeline.plot_overlays(),
        pipeline.dtw_segmentation,
        pipeline.resample_path_equidistant,
        pipeline.arc_length_raw,
        pipeline.trim_region,
        pipeline.dtw_rle,
        partial(pipeline.resample_path_equidistant, factor=1.0, error=1e-5),
        # tap(partial(pipeline.plot_mcp, show=True, save=False, channels=plot_channels)),
        pipeline.backtrack_fraction,
        partial(pipeline.tortuosity, w=9),
        pipeline.central_speed,
        pipeline.arc_length_raw,
        partial(pipeline.turning_angle, w=5),
        partial(pipeline.turning_angle, w=1),
        pipeline.trim_region,
        pipeline.dtw_rle,
        partial(pipeline.resample_path_equidistant, factor=1.0, error=1e-5),
        partial(pipeline.turning_angle, w=1),
        pipeline.central_speed,
        pipeline.arc_length_raw,
        partial(plot.save_strokes_plot(filename_fn=png_gauss, title=f"gauss @ {sigma=}")),
        partial(pipeline.replace_xy, key="gauss:xy"),
        partial(pipeline.gauss_1d, sigma=3.0, f=None),
        conditioning.resample_xy_equidistant,
        pipeline.prune,
        partial(plot.save_strokes_plot(filename_fn=png_raw, title="raw")),
        # tap(lambda s: print(f"{s.dataset} - {s.literal}/{s.stroke_index}"))
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
    # white_list = infer_file_names("出")

    for dataset in datasets:
        directory = f"data/dataset/{dataset}/npy-raw"
        p = compose_pipeline(cuts_target)

        for (dirpath, dirnames, filenames) in os.walk(directory):
            filenames.sort()
            for filename in filenames:
                if not filename.endswith("npy"): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, p, f"{dirpath}/{filename}")
