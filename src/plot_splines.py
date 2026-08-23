#!/usr/bin/env python3
import os
import sys
from signal import signal, SIGINT
import numpy as np
from functools import partial
from svg.path import Path, Move, CubicBezier

from kanji_nn.predef import compose, tap
import kanji_nn.plot as plot
import kanji_nn.io as io


def compose_pipeline():
    return compose(
        plot.paths_plot,
        io.load_splines,
    )


def process_file(dataset, pipeline, filename):
    print(filename)
    pipeline(filename)


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
    # white_list = infer_file_names("子")

    pipeline = compose_pipeline()
    for dataset in datasets:
        directory = f"data/dataset/{dataset}/npy-fitted"

        for (dirpath, dirnames, filenames) in os.walk(directory):
            filenames.sort()
            for filename in filenames:
                if not filename.endswith("npy"): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, pipeline, f"{dirpath}/{filename}")
