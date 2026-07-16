#!/usr/bin/env python3

import csv
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import kanji_nn.metrics as metrics
from kanji_nn.data import Character, Stroke, compose
from kanji_nn.data import find_trim_region, trim_region
from kanji_nn.plot import multi_channel_plot, strokes_plot


def tap(fn):
    def inner(x):
        fn(x)
        return x
    return inner


def plot(stroke):
    channels = ["c_speed", "P_norm", "dP/dt", "ds", "dds", "loc_stness"]
    # channels = ["P_norm", "ds"]

    # figure = multi_channel_plot(stroke, channels, figsize=(18, 8))
    # filename = f"data/dataset/{stroke.dataset}/mcp/{stroke.literal}-{stroke.stroke_index}"
    # plt.savefig(filename)
    # plt.close(figure)
    # plt.show()



composed_metrics = compose(
    # NOTE: after this point stroke lost all props/features.
    # tap(plot),
    find_trim_region,
    metrics.local_straightness,
    metrics.pressure_derivative,
    partial(metrics.tangential_acc, speed_key="c_speed"),
    metrics.vector_acc,
    metrics.curvature,
    metrics.tangent,
    metrics.central_speed,
    metrics.straightness,
    metrics.arc_length,
    metrics.pressure
)


# LITERAL = '左'
# STROKE = 2
LITERAL = None
STROKE = None


def assess(df, row):
    dataset = row['dataset']
    code_point = row['code_point']
    stroke_idx = int(row['stroke_idx'])
    filename = f"data/dataset/{dataset}/npy-raw/{code_point}.npy"

    if STROKE != None and stroke_idx != STROKE:
        return df

    character = Character.of_npy(dataset, filename)

    hce = int(row['head_cut'])
    tce = int(row['tail_cut'])
    rle = tce - hce

    strokes = character.strokes()
    stroke = strokes[stroke_idx]
    stroke = stroke.clone(props={"candidate_head": int(row['head_cut']), "candidate_tail": int(row['tail_cut'])})

    print(stroke.literal, stroke.stroke_index, stroke.stroke_type)
    stroke = composed_metrics(stroke)

    hca, tca = stroke.props['cuts']
    rla = tca - hca

    data = [hce, hca, hca - hce, tce, tca, tca - tce, rle, rla, rla - rle]
    row = pd.DataFrame([data], columns=df.columns)
    return pd.concat([df, row], ignore_index=True)


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    rows = []
    with open("data/analysis-short-samples.csv", mode="r", newline="", encoding="utf-8") as file:
        dict_reader = csv.DictReader(file)
        rows = [row for row in dict_reader]

    columns = ["hce", "hca", "hcd", "tce", "tca", "tcd", "rle", "rla", "rld"]
    df = pd.DataFrame(columns=columns)
    for row in rows:
        if LITERAL != None and row["literal"] != LITERAL:
            continue
        df =  assess(df, row)
