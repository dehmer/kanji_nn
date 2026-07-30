#!/usr/bin/env python3
import os
import sys
from signal import signal, SIGINT
import numpy as np
from functools import partial

from kanji_nn.data import Character, Stroke
from kanji_nn.predef import compose, tap
import kanji_nn.pipeline as pipeline
from kanji_nn.conditioning import join_strokes


plot_channels=["angle"]

def compose_pipeline():
    return compose(
        partial(pipeline.stack_xy, key="savgol:hf:xy"),
        pipeline.savgol_smooth_hf,
        partial(pipeline.resample_path_equidistant, factor=1.0, error=1e-2),
        pipeline.arc_length,
        pipeline.trim_region,
        pipeline.dtw_rle,
        partial(pipeline.resample_path_equidistant, factor=1.0, error=1e-2),
        pipeline.turning_angle,
        pipeline.arc_length,
    )


def process_file(dataset, pipeline, filename):
    character = Character.of_npy(dataset, filename)
    strokes = character.strokes()
    strokes = [pipeline(s) for s in strokes]
    strokes = [s.raw for s in strokes]
    raw = join_strokes(strokes)
    np.save(f'data/dataset/{dataset}/npy-trimmed/{character.code_point}.npy', raw)


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
    # white_list = infer_file_names("そぬねみむ")
    # white_list = infer_file_names("む")

    for dataset in datasets:
        directory = f'data/dataset/{dataset}/npy-raw'
        p = compose_pipeline()

        for (dirpath, dirnames, filenames) in os.walk(directory):
            for filename in filenames:
                if not filename.endswith('npy'): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, p, f'{dirpath}/{filename}')
