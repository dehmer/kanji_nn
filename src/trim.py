#!/usr/bin/env python3
import os
import sys
from signal import signal, SIGINT
import numpy as np
from functools import partial
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from kanji_nn.data import compose, tap, Character, Stroke
import kanji_nn.metrics as metrics
import kanji_nn.svg as svg
from kanji_nn.plot import strokes_plot


def resample_xy(stroke, factor=1.0):
    n_out = round(stroke.n_points * factor)
    n_out = max(n_out, 2)

    s = stroke.features['raw:s']
    samples = np.linspace(0.0, s[-1], n_out)
    x = np.interp(samples, s, stroke.x)
    y = np.interp(samples, s, stroke.y)
    xy = np.column_stack([x, y])

    return stroke.clone(props={"resampled:xy": xy})


def resample_path_parametric(stroke, error=1e-5):
    """
    Resamples the path using raw parametric steps.
    NOTE: Vertices will NOT be perfectly equidistant by distance.
    Sampling happens in the parametric domain, not the spatial domain.
    This parameterization is non-linear with respect to distance for
    most complex curves.
    """
    path = stroke.sticky["path"]
    ts = np.linspace(0.0, 1.0, stroke.n_points)

    # Pull points directly from the main path object
    vertices = np.zeros((stroke.n_points, 2), dtype=np.float64)
    for i, t in enumerate(ts):
        raw_point = path.point(t, error)
        vertices[i, 0] = raw_point.real
        vertices[i, 1] = raw_point.imag

    return stroke.clone(props={"path:xy": vertices})


def resample_path_equidistant(stroke, error=1e-5):
    path = stroke.sticky["path"]
    xy = stroke.props["resampled:xy"]
    raw_s = stroke.features["raw:s"]
    max_ds = raw_s[-1] / stroke.n_points

    xys = svg.resample_equidistant(path, len(xy), error=error)
    return stroke.clone(props={"path:xys": xys})

def rle(stroke):
    resampled_xy = stroke.props["resampled:xy"]
    path_xys = stroke.props["path:xys"]
    print(len(resampled_xy), len(path_xys))

    A = 0 # path column index: (handwritten) stroke
    B = 1 # path column index: reference
    # True/1:  consecutive A-indices hit same B-index => dirty
    # False/0: strict monotonic advancement           => clean

    radius = 1
    distance, path = fastdtw(resampled_xy, path_xys[:, -1], radius=radius, dist=euclidean)
    path = np.asarray(path)
    mask = np.diff(path[:, B]) == 0
    edges = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1, len(mask)]
    signature = np.asarray(list(zip(mask[edges[:-1]], np.diff(edges))))
    print(f"{stroke.literal}/{stroke.stroke_index}: ", "signature", distance, "\n", signature)

    return stroke

def compose_pipeline():
    return compose(
        rle,
        partial(resample_path_equidistant, error=1e-2),
        partial(resample_xy, factor=1.0),
        metrics.arc_length_raw,
        # tap(lambda s: print(s)),
    )


def process_file(dataset, pipeline, filename):

    # reflect, nearest, mirror
    char = Character.of_npy(dataset, filename)
    strokes = char.strokes()
    trimmed = [pipeline(s) for s in strokes]

    # xy = [s.xy for s in trimmed]
    xy = [s.props["resampled:xy"] for s in trimmed]
    # xy = [s.props["path:xys"] for s in trimmed]
    strokes_plot.show(xy, alpha=0.0)


def literal_to_hex(literal):
    return f'{ord(literal):x}'.upper()


def infer_file_names(literals):
    return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())
    datasets = ['katakana_47', 'hiragana_46', 'kanken-10_80']

    white_list = []
    # white_list = infer_file_names("ふらなまね虫")

    for dataset in datasets:
        directory = f'data/dataset/{dataset}/npy-raw'
        pipeline = compose_pipeline()

        for (dirpath, dirnames, filenames) in os.walk(directory):
            for filename in filenames:
                if not filename.endswith('npy'): continue
                if white_list and filename not in white_list: continue
                process_file(dataset, pipeline, f'{dirpath}/{filename}')
