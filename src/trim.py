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


def dump(stroke):
    print(f"{stroke.dataset},{stroke.key},{stroke.props["cuts"][0]},{stroke.props["cuts"][1]}")


def compare(cuts_target):
    max = 0.0
    def inner(stroke):
        head = f"{stroke.literal}/{stroke.stroke_index} - "

        if not stroke.key in cuts_target:
            return

        nonlocal max
        t = stroke.t - stroke.t[0]
        a = stroke.props["cuts"]
        e = cuts_target[stroke.key]
        d = (abs(e[0] - a[0]), abs(e[1] - a[1]))
        if d != (0, 0):
            print(head, f"expected: {e}, actual: {a}, difference: {d}")

    return inner


def inject_cuts_target(stroke, cuts_target):
    return stroke.clone(props={"cuts_target": cuts_target[stroke.key]})


plot_channels=[
    "angle:w=1"
]

png_trimmed = lambda s: f"data/dataset/{s.dataset}/png-trimmed/{s.code_point}"


def compose_pipeline(cuts_target):
    sigma = 1.0     # Gauss 1D Filter
    return compose(
        plot.show_strokes_plot(lambda s: s.features["xy"]),
        plot.save_strokes_plot(filename_fn=png_trimmed),
        io.save_npy("npy-trimmed"),
        pipeline.trim_region,
        tap(partial(plot.show_mcp_plot, show=True, save=False, channels=plot_channels)),
        # tap(dump),
        # tap(compare(cuts_target)),
        pipeline.dtw_rle,
        partial(pipeline.resample_path_equidistant, factor=1.0, error=1e-5),
        partial(pipeline.turning_angle, w=1),
        pipeline.arc_length_raw,
        partial(pipeline.replace_xy, key="gauss:xy"),
        partial(pipeline.gauss_1d, sigma=sigma, f=None),
        pipeline.prune,
        pipeline.split_raw,
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
    # white_list = infer_file_names("林")

    for dataset in datasets:
        directory = f"data/dataset/{dataset}/npy-raw"
        p = compose_pipeline(cuts_target)

        for (dirpath, dirnames, filenames) in os.walk(directory):
            filenames.sort()
            for filename in filenames:
                if not filename.endswith("npy"): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, p, f"{dirpath}/{filename}")
