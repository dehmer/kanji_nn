#!/usr/bin/env python3
import os
import sys
from signal import signal, SIGINT
import numpy as np
import csv
import math

from kanji_nn.data import Character
from kanji_nn.pipelines import post_pipeline
from kanji_nn.cli import *


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())

    datasets = [
        # "katakana_47",
        # "hiragana_46",
        # "kanken-10_80",
        "kanken-09_160",
    ]

    white_list = []
    # white_list = infer_file_names("子")

    pipeline = post_pipeline()
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
