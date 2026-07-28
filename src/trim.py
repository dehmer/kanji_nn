#!/usr/bin/env python3
import os
import sys
from signal import signal, SIGINT
import numpy as np
from functools import partial
from itertools import accumulate
from dtw import dtw, asymmetric
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from kanji_nn.data import Character, Stroke
from kanji_nn.predef import compose, tap
import kanji_nn.pipeline as pipeline
from kanji_nn.plot import strokes_plot
from kanji_nn.data import plot_mcp # TODO: move


def resample_xy(stroke, factor=1.0):
    n_out = round(stroke.n_points * factor)
    n_out = max(n_out, 2)

    s = stroke.features['raw:s']
    samples = np.linspace(0.0, s[-1], n_out)
    x = np.interp(samples, s, stroke.x)
    y = np.interp(samples, s, stroke.y)
    xy = np.column_stack([x, y])

    return stroke.clone(props={"resampled:xy": xy})


def rle(stroke):
    """
    Wording:
        * query, reference, (optomized) warping path
        * one-to-many "stagnation" points
    """

    query = stroke.xy
    path_xys = stroke.props["path:xys"]
    reference = path_xys[:, :-1]

    W = dtw(
        query,
        reference,
        dist_method="euclidean",
        step_pattern=asymmetric,  # Crucial for open-ended alignment
        open_begin=True,          # Relaxes the starting boundary condition
        open_end=True,            # Relaxes the ending boundary condition
        keep_internals=False
    )


    # Create "stroke signature" with respective run lengths
    # and cumulative run length sums for T/F groups:
    mask = np.diff(W.index2) == 0
    edges = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1, len(mask)]
    s = list(zip(mask[edges[:-1]], np.diff(edges)))

    s = [
        (tag, rl, W.index1[cs]) for (tag, rl), cs in zip(
            s,
            accumulate(length for _, length in s)
        )
    ]

    # print(f"{stroke.literal}/{stroke.stroke_index}", stroke.n_points)

    i, j = 0, len(s) - 1 # [i, j]

    # unconditional. cut dirty at 0 and n-1
    if s[i][0]: i = i + 1
    if s[j][0]: j = j - 1

    # max_gap = 2
    # # jump small clean gaps (length < max_gap)
    # if s[i][1] < max_gap: i = i + 2
    # if s[j][1] < max_gap: j = j - 2

    head_cut = 0 if i == 0 else s[i-1][2] + 1 # inclusive
    tail_cut = s[j][2] + 1                    # exclusive
    cuts = (head_cut, tail_cut)

    return stroke.clone(props={"cuts": cuts})


plot_channels=["angle"]

def compose_pipeline():
    return compose(
        pipeline.trim_region,
        # tap(partial(plot_mcp, show=True, save=False, channels=plot_channels)),
        rle,
        partial(pipeline.resample_path_equidistant, factor=1.0, error=1e-2),
        pipeline.turning_angle,
        pipeline.arc_length,
    )


def process_file(dataset, pipeline, filename):

    # reflect, nearest, mirror
    char = Character.of_npy(dataset, filename)
    strokes = char.strokes()
    raw_xy = [s.xy for s in strokes]
    trimmed = [pipeline(s) for s in strokes]

    # xy = [s.xy for s in trimmed]
    trimmed_xy = [s.xy for s in trimmed]
    # path_xy = [s.props["path:xys"][:, :-1] for s in trimmed]
    # strokes_plot.show(trimmed_xy, alpha=0.0)

    # strokes_plot.overlays([raw_xy, trimmed_xy], styles=[
    #     {"color": "red", "linewidth": 2.0, "alpha": 1.0},
    #     {"color": "black", "linewidth": 2.0, "alpha": 1.0},
    # ])

    plot_filename = f'data/dataset/{dataset}/png-post/{char.code_point}'
    strokes_plot.save(plot_filename, trimmed_xy, alpha=0.1)



def literal_to_hex(literal):
    return f'{ord(literal):x}'.upper()


def infer_file_names(literals):
    return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())
    datasets = ['katakana_47', 'hiragana_46', 'kanken-10_80']

    white_list = []
    # white_list = infer_file_names("ふらなまね虫")
    # white_list = infer_file_names("虫")

    for dataset in datasets:
        directory = f'data/dataset/{dataset}/npy-raw'
        p = compose_pipeline()

        for (dirpath, dirnames, filenames) in os.walk(directory):
            for filename in filenames:
                if not filename.endswith('npy'): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, p, f'{dirpath}/{filename}')
