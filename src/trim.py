#!/usr/bin/env python3
import os
import sys
from signal import signal, SIGINT
import numpy as np
from functools import partial
import csv
import math

from kanji_nn.data import Character
from kanji_nn.pipelines.trim_pipeline import trim_pipeline
from kanji_nn.cli import *


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
        # "katakana_47",
        # "hiragana_46",
        # "kanken-10_80",
        "kanken-09_160",
    ]

    cuts_target = load_cuts("data/cuts-baseline.csv")

    white_list = []
    # white_list = infer_file_names("林")

    pipeline = trim_pipeline(cuts_target)
    for dataset in datasets:
        directory = f"data/dataset/{dataset}/npy-raw"

        for (dirpath, dirnames, filenames) in os.walk(directory):
            filenames.sort()
            for filename in filenames:
                if not filename.endswith("npy"): continue
                if white_list and filename not in white_list: continue

                filename = f"{dirpath}/{filename}"
                character = Character.of_npy(dataset, filename)
                strokes = character.strokes()
                [pipeline(stroke) for stroke in strokes]
