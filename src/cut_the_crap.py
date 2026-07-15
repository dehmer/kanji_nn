#!/usr/bin/env python3

import csv
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from kanji_nn.data import Character, Stroke, compose
import kanji_nn.metrics as metrics
from kanji_nn.plot import multi_channel_plot


def do_the_cutting(stroke):
    region = stroke.props["cuts"]
    return stroke.trim(region)


def tap(fn):
    def inner(x):
        fn(x)
        return x
    return inner


def plot(stroke):
    channels = ["c_speed", "P_norm", "dP/dt", "ds", "loc_stness"]
    # channels = ["P_norm", "ds"]

    figure = multi_channel_plot(stroke, channels, figsize=(18, 8))
    # filename = f"data/dataset/{stroke.dataset}/mcp/{stroke.literal}-{stroke.stroke_index}"
    # plt.savefig(filename)
    # plt.close(figure)
    plt.show()

default_detector = lambda stroke: stroke.clone(props = {"cuts": (0, stroke.n_points)})

def CJK_STROKE_H(stroke):
    r"""
    Legend:
        abc(i/j[f]) = rising edge on abc between i,j.
        abc(i\j[f]) = falling edge.
        f = '*' both endpoints viable,
        f = <idx> only that endpoint viable.

    Note: all boundaries/indices are strictly inclusive: [a, b].

    左/0 - general:
    ds, c_speed:
        Unimodal, symmetric, platykurtic, bracketing entire clean region,
        roughly 1/3 rise -> 1/3 plateau -> 1/3 decline,
        similar positive and negative slopes.
    P_norm:
        Steady climb in dirty region 23% - shy of 87%,
        then in clean region after initial hump,
        prolonged plateau until decline,
        declines only way after clean region (+66ms).
    loc_stness:
        Oscillating heavily (1-0-1) at dirty head and
        one negative spike at dirty tail,
        inner spikes frame clean region well.

    左/0 - head:
    candidates 4 [15:18]: seemingly the same point
        pressure_norm(15:18) = (0.75 - 0.91)
        dP/dt(15:18) > 0
        ds(17/18[*])
        loc_stness(15\16[*])
        loc_stness(16/17[*])
    """

    # head cut detector:
    t = stroke.t
    ds = stroke.features["ds"]

    cs_mask = ds >= np.percentile(ds, 85)
    t_centroid = np.average(t[cs_mask], weights=ds[cs_mask])

    # top/left part:     < centroid_idx
    # bottom/right part: > centroid_idx
    centroid_idx = np.argmin(np.abs(t - t_centroid))





    # tail cut detector:

    # exit()




    # TODO: should be fun to work on this first.
    return default_detector(stroke)

detectors = {
    "CJK STROKE H": CJK_STROKE_H
}

# scaffolding
def do_the_magic_trick(stroke):
    _, _, name = stroke.stroke_type
    detector = detectors.get(name, default_detector)
    return detector(stroke)

composed_metrics = compose(
    # NOTE: after this point stroke lost all props/features.
    # do_the_cutting,
    tap(plot),
    do_the_magic_trick,
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

def assess(df, row):
    dataset = row['dataset']
    code_point = row['code_point']
    stroke_idx = int(row['stroke_idx'])
    filename = f"data/dataset/{dataset}/npy-raw/{code_point}.npy"


    TARGET_STROKE = 0
    if not stroke_idx == TARGET_STROKE:
        return df


    character = Character.of_npy(dataset, filename)

    hce = int(row['head_cut'])
    tce = int(row['tail_cut'])
    rle = tce - hce

    strokes = character.strokes()
    stroke = strokes[stroke_idx]
    print(stroke.literal, stroke.stroke_index, stroke.stroke_type)
    stroke = composed_metrics(stroke)

    hca, tca = stroke.props['cuts']
    rla = tca - hca

    data = [hce, hca, hca - hce, tce, tca, tca - tce, rle, rla, rla - rle]
    row = pd.DataFrame([data], columns=df.columns)
    return pd.concat([df, row], ignore_index=True)


if __name__ == "__main__":
    rows = []
    with open("data/analysis-short-samples.csv", mode="r", newline="", encoding="utf-8") as file:
        dict_reader = csv.DictReader(file)
        rows = [row for row in dict_reader]

    columns = ["hce", "hca", "hcd", "tce", "tca", "tcd", "rle", "rla", "rld"]
    df = pd.DataFrame(columns=columns)
    for row in rows:
        if row["literal"] != '左':
            continue
        df =  assess(df, row)

    # print(df)