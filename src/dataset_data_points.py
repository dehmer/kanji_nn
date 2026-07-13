#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd
from kanji_nn.conditioning import split_strokes


def extract_code_point(filename):
    match = re.search(r"(U\+[0-9A-F]{4,5})", filename)
    if not match:
        raise Exception(f'cp: invalid format in {filename}')
    return match.group(1)


def cp_to_chr(cp):
    return chr(int(cp[2:], 16))


def literal_to_hex(literal):
    return f'{ord(literal):x}'.upper()


def infer_file_names(literals):
    return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]


def stroke_data_frames(filename, dataset):
    """
    One raw, full-resolution DataFrame per stroke (all points, columns
    [timestamp, x, y, pressure]), tagged with dataset/char/stroke_idx.
    This is the substrate for both 1-tier summaries (below) and any
    heavier 2-tier metric work later (curvature, smoothing, etc.) -
    kept un-collapsed on purpose.
    """
    code_point = extract_code_point(filename)
    literal = cp_to_chr(code_point)

    raw = np.load(filename)
    raw = raw if raw.shape[1] == 5 else raw[:, (0, 1, 2, 3, 6)]
    strokes = split_strokes(raw)

    columns = ['timestamp', 'x', 'y', 'pressure']

    dfs = []
    for stroke_idx, stroke in enumerate(strokes):
        stroke = stroke.copy()  # avoid mutating whatever split_strokes handed back
        stroke[:, 0] -= stroke[0, 0]  # normalize timestamp to 0ms per stroke

        df = pd.DataFrame(stroke, columns=columns)
        # DataFrame.insert(position, name, value) adds a column at a fixed
        # position. A scalar value (e.g. the string `dataset`) gets broadcast
        # to every row - pandas is fine mixing string/float columns in one table.
        df.insert(0, 'dataset', dataset)
        df.insert(1, 'char', literal)
        df.insert(2, 'stroke_idx', stroke_idx)
        dfs.append(df)

    return dfs


def stroke_summary(df):
    """
    Reduce one raw per-stroke DataFrame to a single 1-tier summary row
    (a pandas Series - think of it as one row's worth of named values).
    """
    # .diff() gives consecutive differences (row i - row i-1); the first
    # row is NaN since there's no predecessor. np.hypot(dx, dy) is the
    # per-segment Euclidean step length; .sum() skips NaN by default,
    # so the leading NaN just drops out of the total.
    dx = df['x'].diff()
    dy = df['y'].diff()
    arc_length = np.hypot(dx, dy).sum()

    return pd.Series({
        'dataset': df['dataset'].iloc[0],
        'char': df['char'].iloc[0],
        'stroke_idx': df['stroke_idx'].iloc[0],
        'n_points': len(df),
        'duration_ms': df['timestamp'].iloc[-1] - df['timestamp'].iloc[0],
        'arc_length': arc_length,
        'pressure_max': df['pressure'].max(),
        'pressure_std': df['pressure'].std(),  # sample std (ddof=1) by default
    })


def load_dataset(dataset, in_dir, white_list=None):
    white_list = white_list or []
    dfs = []
    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        for filename in filenames:
            if not filename.endswith('npy'):
                continue
            if white_list and filename not in white_list:
                continue
            dfs.extend(stroke_data_frames(f'{dirpath}/{filename}', dataset))
    return dfs


if __name__ == "__main__":
    datasets = {
        'hiragana_48': 'data/dataset/hiragana_48/npy-raw',
        'katakana_49': 'data/dataset/katakana_49/npy-raw',
        'kanken-10_80': 'data/dataset/kanken-10_80/npy-raw',
    }

    # white_list = infer_file_names(['目'])  # restrict to specific characters if needed
    white_list = []

    stroke_dfs = []
    for dataset, in_dir in datasets.items():
        stroke_dfs.extend(load_dataset(dataset, in_dir, white_list))

    # One summary Series per stroke -> stack into a single DataFrame.
    # pd.DataFrame(list_of_series) treats each Series as a row.
    summary = pd.DataFrame([stroke_summary(df) for df in stroke_dfs])

    # groupby('dataset') splits the table into per-dataset groups;
    # .describe() gives count/mean/std/min/25%/50%/75%/max per numeric column,
    # per group - exactly the 1-tier stats requested (min arc length,
    # pressure max/std, duration min/max/std) plus a few extras for free.
    per_dataset = summary.groupby('dataset').describe()

    summary.to_csv('stroke_summary.csv', index=False)
    per_dataset.to_csv('stroke_summary_by_dataset.csv')
