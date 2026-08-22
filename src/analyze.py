#!/usr/bin/env python3
import os
import sys
from signal import signal, SIGINT
import numpy as np
from functools import partial
import csv
import math

from kanji_nn.data import Character, Stroke
from kanji_nn.predef import compose, tap
import kanji_nn.analysis as analysis
import kanji_nn.io as io
import kanji_nn.plot as plot
import kanji_nn.analysis as analysis
import kanji_nn.metrics as metrics
import kanji_nn.conditioning as conditioning


plot_channels=[
    "pressure",
    "dP/dt",
    "raw:speed:backward",
    "raw:speed:central",
    "gauss:dθ/ds:abs",
    "raw:stness:loc",
    "raw:speed:forward",
    "raw:stness",
    # "angle:w=1:abs",
    # "raw:speed:central",
]


def compose_pipeline():
    return compose(
        tap(partial(plot.show_mcp_plot, show=True, save=False, channels=plot_channels)),
        # analysis.density(),
        # metrics.arc_length_raw,
        metrics.local_straightness,
        metrics.straightness,
        metrics.curvature,
        metrics.tangent,
        metrics.arc_length_gauss,
        metrics.arc_length_raw,
        metrics.central_speed,
        metrics.backward_speed,
        metrics.forward_speed,
        metrics.pressure_derivative,
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


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())

    datasets = [
        "katakana_47",
        "hiragana_46",
        "kanken-10_80",
    ]

    white_list = []
    # white_list = infer_file_names("そ")

    for dataset in datasets:
        directory = f"data/dataset/{dataset}/npy-raw"
        p = compose_pipeline()

        for (dirpath, dirnames, filenames) in os.walk(directory):
            filenames.sort()
            for filename in filenames:
                if not filename.endswith("npy"): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, p, f"{dirpath}/{filename}")
