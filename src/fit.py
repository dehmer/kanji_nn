#!/usr/bin/env python3
import os
import sys
from signal import signal, SIGINT
import numpy as np
from functools import partial

from kanji_nn.data import Character, Stroke
from kanji_nn.predef import compose, tap
import kanji_nn.bezier as bezier
import kanji_nn.io as io
import kanji_nn.plot as plot
import kanji_nn.conditioning as conditioning
import kanji_nn.metrics as metrics
import kanji_nn.bezier as bezier


fitted_path = lambda s: s.props["fitted"]
xys         = lambda s: s.props["path:xys"][:, :-1]
png_fitted  = lambda s: f"data/dataset/{s.dataset}/png-fitted/{s.code_point}"


def compose_pipeline():
    ds = 0.006      # slightly below minimum of all strokes
    alpha = 0.1
    epsilon = 3e-4  # RDP
    maxError = 5e-4 # Schneider's Algorithm

    return compose(
        io.save_splines("npy-fitted"),
        partial(plot.save_strokes_plot(filename_fn=png_fitted, xy_fn=xys, alpha=alpha)),
        partial(bezier.resample_fixed_distance, path_fn=fitted_path, ds=ds),
        partial(bezier.schneider, maxError=maxError),

        partial(conditioning.simplify_rdp, epsilon=epsilon),
        conditioning.split_raw,
        tap(lambda s: print(f"{s.dataset} - {s.literal}/{s.stroke_index}"))
    )


def process_file(dataset, pipeline, filename):
    character = Character.of_npy(dataset, filename)
    strokes = character.strokes()
    strokes = [pipeline(s) for s in strokes]


def literal_to_hex(literal):
    return f'{ord(literal):x}'.upper()


def infer_file_names(literals):
    return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())
    datasets = [
        'katakana_47',
        'hiragana_46',
        'kanken-10_80',
    ]

    white_list = []
    # white_list = infer_file_names("出")

    pipeline = compose_pipeline()
    for dataset in datasets:
        directory = f'data/dataset/{dataset}/npy-trimmed'

        for (dirpath, dirnames, filenames) in os.walk(directory):
            filenames.sort()
            for filename in filenames:
                if not filename.endswith('npy'): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, pipeline, f'{dirpath}/{filename}')
