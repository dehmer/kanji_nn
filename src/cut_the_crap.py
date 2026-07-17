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

bimodal = [
    "CJK STROKE HG",
    "CJK STROKE WG",
    "CJK STROKE N",
    "CJK STROKE HZ",
    "CJK STROKE SG",
    "CJK STROKE SWG",
]

def plot(stroke):
    # if stroke.stroke_type[2] not in bimodal:
    #     return stroke

    channels = ["P_norm", "dP/dt", "ds", "stness", "loc_stness"]
    # channels = list(stroke.features.keys())

    figure = multi_channel_plot(stroke, channels, figsize=(18, 8))
    plt.show()
    # filename = f"data/dataset/{stroke.dataset}/mcp/{stroke.literal}-{stroke.stroke_index}"
    # plt.savefig(filename)
    # plt.close(figure)



composed_metrics = compose(
    # NOTE: after this point stroke lost all props/features.
    # tap(plot),
    find_trim_region,
    metrics.local_straightness,
    partial(metrics.tangential_acc, speed_key="c_speed"),
    metrics.vector_acc,
    metrics.curvature,
    metrics.tangent,
    metrics.central_speed,
    metrics.straightness,
    metrics.arc_length,
    metrics.pressure_derivative,
    metrics.pressure
)


# LITERAL = "足"
# STROKE = 2
LITERAL = None
STROKE = None


def assess(df, row):
    dataset = row["dataset"]
    code_point = row["code_point"]
    stroke_idx = int(row["stroke_idx"])
    filename = f"data/dataset/{dataset}/npy-raw/{code_point}.npy"

    if STROKE != None and stroke_idx != STROKE:
        return df

    character = Character.of_npy(dataset, filename)

    hce = int(row["head_cut"])
    tce = int(row["tail_cut"])
    rle = tce - hce

    strokes = character.strokes()
    stroke = composed_metrics(strokes[stroke_idx])

    if stroke.stroke_type[2] == "CJK STROKE H":
        cuts = stroke.props["cuts"]
        head_cut = int(row["head_cut"])
        tail_cut = int(row["tail_cut"])
        error = (head_cut - cuts[0], tail_cut - cuts[1])
        print(stroke.literal, stroke.stroke_index, stroke.stroke_type, f"head_cut={head_cut}, tail_cut={tail_cut}, error={error}")
    else:
        # print(stroke.literal, stroke.stroke_index, stroke.stroke_type)
        pass

    hca, tca = stroke.props["cuts"]
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
