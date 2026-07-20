#!/usr/bin/env python3

import csv
from functools import partial
import numpy as np

import kanji_nn.metrics as metrics
from kanji_nn.data import Character, Stroke, compose
from kanji_nn.data import find_trim_region, tap, plot_mcp
from kanji_nn.cpd import optimize_pipeline, compute_combined_signal
from kanji_nn.cli import *


# cpd_channels = ["P_inv", "c_speed", "K"]
cpd_channels = ["loc_stness", "K"]

composed_metrics = compose(
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

    strokes = [composed_metrics(stroke) for stroke in strokes]
    def S(stroke, weights):
        return compute_combined_signal(stroke, cpd_channels, weights)

    best_params, best_mae = optimize_pipeline(strokes, S, len(cpd_channels))
    weights, window_pct = best_params["weights"], best_params["window_pct"]
    strokes = [s.clone(props={"weights": weights, "window_pct": window_pct}) for s in strokes]

    print("best_mae", best_mae)
    print("channels", cpd_channels)
    print("weights", weights)
    print("window_pct", window_pct)
