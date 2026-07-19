#!/usr/bin/env python3

import csv
from functools import partial
import numpy as np

import kanji_nn.metrics as metrics
from kanji_nn.data import Character, Stroke, compose
from kanji_nn.data import find_trim_region, tap, plot_mcp
from kanji_nn.cpd import optimize_pipeline


plot_channels=["P", "ds", "c_speed", "θ", "dθ/ds", "K"]
cpd_channels, cpd_weights = ["P_inv", "c_speed", "K"], [0.5, 0.35, 0.15]

composed_metrics = compose(
    # tap(partial(plot_mcp, channels=plot_channels, show=False)),
    find_trim_region,
    partial(metrics.cpd_signal, channels=cpd_channels, weights=cpd_weights),
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
    dataset = row['dataset']
    literal = row["literal"]
    code_point = f"U+{ord(literal):04X}"
    return f"{dataset}:{code_point}"


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
        strokes[stroke_idx] = stroke.clone(props={"cuts": cuts})

    return strokes_dict


if __name__ == "__main__":
    rows = []
    with open("data/expected-cuts.csv", mode="r", newline="", encoding="utf-8") as file:
        dict_reader = csv.DictReader(file)
        rows = [row for row in dict_reader]

    keys = distinct_chars(rows)
    strokes_dict = load_strokes(keys)
    strokes_dict = preload_cuts(strokes_dict, rows)
    strokes = [stroke for strokes in strokes_dict.values() for stroke in strokes]
    strokes = [stroke for stroke in strokes if stroke.stroke_type[2] == "HZ"]
    # strokes = [stroke for stroke in strokes if stroke.literal == '石']
    strokes = sorted(strokes, key = lambda s: s.literal)

    # strokes = [composed_metrics(stroke) for stroke in strokes]
    # best_params, best_mae = optimize_pipeline(strokes)
    # print(best_params, best_mae)
    # exit()


    for stroke in strokes:
        expected = stroke.props["cuts"]
        stroke = composed_metrics(stroke)
        actual = stroke.props["cuts"]
        error = (actual[0] - expected[0], actual[1] - expected[1])
        print(stroke.literal, stroke.stroke_index, stroke.stroke_type, f"expected={expected}\tactual={actual}\terror={error}")
