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
    "gauss:θ",
    "gauss:txy"
]


def compose_pipeline():
    return compose(
        tap(partial(pipeline.plot_mcp, show=True, save=False, channels=plot_channels)),
        pipeline.central_speed,
        partial(pipeline.turning_angle, w=1),
        pipeline.curvature,
        pipeline.tangent,
        pipeline.arc_length,
        pipeline.cleanup_clusters,
        tap(partial(pipeline.plot_mcp, show=True, save=False, channels=plot_channels)),
        pipeline.detect_clusters,
        pipeline.central_speed,
        partial(pipeline.turning_angle, w=1),
        pipeline.curvature,
        pipeline.tangent,
        pipeline.arc_length,
        tap(lambda s: print(f"{s.dataset} - {s.literal}/{s.stroke_index}"))
    )


def process_file(dataset, pipeline, filename):
    character = Character.of_npy(dataset, filename)
    strokes = character.strokes()
    strokes = [pipeline(s) for s in strokes]
    xy = [s.xy for s in strokes]
    strokes_plot.show(xy, alpha=0.0)

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

    complex_kanken = "七中九五先円出力口右名四夕女子字学小山手日早月村気水田男町白百目石空竹糸花草虫見貝赤足車雨青音"
    complex_hiragana = "おかきせそたなにねはほみりるれろわいうえけこさちてひふむやゆらをん"
    complex_katakana = "アウオカクコスセタヌネヒフホマムヤユヨラルレロワ"
    backtrace = "えきけこさせそたちてなにねひみむゆらるろをアオカネムラワ中五円出夕子学小山手早村空糸花草見青"
    angle_prominence_pi_third = "いうえおかきくけこさせそたちてなにぬねはひほまみむめゆらるれろわをんアウオカコスセツヌネホマムヤラルレロワヲー上中九五人先円出力名夕女子字学小山川手日早月村気水町空竹糸花草見赤足車雨青"

    complex = complex_kanken + complex_hiragana + complex_katakana

    white_list = []
    # white_list = infer_file_names(complex)
    # white_list = infer_file_names(backtrace)
    # white_list = infer_file_names(angle_prominence_pi_third)
    white_list = infer_file_names("ア")

    for dataset in datasets:
        directory = f'data/dataset/{dataset}/npy-trimmed'
        p = compose_pipeline()

        for (dirpath, dirnames, filenames) in os.walk(directory):
            for filename in filenames:
                if not filename.endswith('npy'): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, p, f'{dirpath}/{filename}')
