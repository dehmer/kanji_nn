#!/usr/bin/env python3

import csv
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import kanji_nn.metrics as metrics
from kanji_nn.data import Character, Stroke, compose
from kanji_nn.data import find_trim_region, trim_region
from kanji_nn.plot import multi_channel_plot, strokes_plot
import kanji_nn.cpd as cpd


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
    if stroke.stroke_type[2] not in bimodal:
        return stroke

    channels = ["P_norm", "dP/dt", "dP", "ds", "c_speed", "at", "stness", "loc_stness"]
    figure = multi_channel_plot(stroke, channels, figsize=(18, 8))
    filename = f"data/dataset/{stroke.dataset}/mcp/{stroke.literal}-{stroke.stroke_index}"
    plt.savefig(filename)
    # plt.show()
    plt.close(figure)


composed_metrics = compose(
    # NOTE: after this point stroke lost all props/features.
    # tap(plot),
    # find_trim_region,
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


def char_key(row):
    return f"{row['dataset']}:{row["code_point"]}"

def distinct_chars(rows):
    keys = [char_key(row) for row in rows]
    return set(keys)

def load_strokes(keys):
    chars = dict()
    for key in keys:
        dataset, code_point = tuple(key.split(':'))
        filename = f"data/dataset/{dataset}/npy-raw/{code_point}.npy"
        character = Character.of_npy(dataset, filename)
        chars[key] = character.strokes()
    return chars

def preload_cuts(strokes_dict, rows):
    for row in rows:
        stroke_idx = int(row["stroke_idx"])
        cuts = (int(row["head_cut"]), int(row["tail_cut"]))
        strokes = strokes_dict[char_key(row)]
        stroke = strokes[stroke_idx]
        stroke = composed_metrics(stroke)
        strokes[stroke_idx] = stroke.clone(props={"cuts": cuts})

    return strokes_dict

def find_change_point(stroke, w1, w2, w3, window_size=6):
    window_pct = 0.0
    S = cpd.compute_combined_signal(stroke, w1, w2, w3)
    head_cut, tail_cut = stroke.props["cuts"]
    # idx = cpd.find_change_point_adaptive(S, window_pct)
    idx = cpd.find_change_point(S, window_size=window_size)
    print(stroke.literal, stroke.stroke_index, stroke.stroke_type[2], idx, tail_cut, tail_cut - idx)

bimodal = [
    "CJK STROKE HG",
    "CJK STROKE WG",
    "CJK STROKE N",
    "CJK STROKE HZ",
    "CJK STROKE SG",
    "CJK STROKE SWG",
]

if __name__ == "__main__":
    rows = []
    with open("data/analysis-short-samples.csv", mode="r", newline="", encoding="utf-8") as file:
        dict_reader = csv.DictReader(file)
        rows = [row for row in dict_reader]

    keys = distinct_chars(rows)
    strokes_dict = load_strokes(keys)
    strokes_dict = preload_cuts(strokes_dict, rows)
    strokes = [stroke for strokes in strokes_dict.values() for stroke in strokes]
    strokes = [stroke for stroke in strokes if stroke.stroke_type[2] in bimodal]

    # best_params, best_mae = cpd.optimize_pipeline(strokes)
    # print(best_params, best_mae)

    best_weights, best_mae = cpd.optimize_weights(strokes)
    print(best_weights, best_mae)
    w1, w2, w3 = best_weights

    for stroke in strokes:
        find_change_point(stroke, w1, w2, w3)
