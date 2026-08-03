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
from kanji_nn.plot import strokes_plot, paths_plot


plot_channels=[
    "angle:w=1:abs",
    "raw:speed:central",
    "gauss:tx", "gauss:ty",
    "gauss:dθ/ds:abs"]


def compose_pipeline():
    return compose(
        tap(partial(pipeline.plot_mcp, show=True, save=False, channels=plot_channels)),
        pipeline.detect_tracebacks,
        pipeline.central_speed,
        partial(pipeline.turning_angle, w=1),
        pipeline.curvature,
        pipeline.tangent,
        pipeline.arc_length,
    )


def process_file(dataset, pipeline, filename):
    character = Character.of_npy(dataset, filename)
    strokes = character.strokes()
    strokes = [pipeline(s) for s in strokes]
    raw = [s.raw for s in strokes]
    raw = join_strokes(raw)
    paths = [s.sticky["path"] for s in strokes]
    # paths_plot(paths, show_badges=False)
    # np.save(f'data/dataset/{dataset}/npy-trimmed/{character.code_point}.npy', raw)


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
    # white_list = infer_file_names("アセぬむ虫気")
    white_list = infer_file_names("コネルヤレムモロヲカオアこおやるれそういたんゆをけさろくきむてひまなにみえふらね車円小青五貝学早森見足草町夕水山空上立白耳赤村気出糸日字休花竹雨玉子先手月")
    white_list = infer_file_names("ヤモヲオおやれそたさきてまななふ車小青学足夕水空立白気出糸日休花竹玉子先")

    for dataset in datasets:
        directory = f'data/dataset/{dataset}/npy-trimmed'
        p = compose_pipeline()

        for (dirpath, dirnames, filenames) in os.walk(directory):
            for filename in filenames:
                if not filename.endswith('npy'): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, p, f'{dirpath}/{filename}')
