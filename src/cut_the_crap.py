#!/usr/bin/env python3

import csv
from functools import partial
import numpy as np

import kanji_nn.metrics as metrics
from kanji_nn.data import Character, Stroke, compose
from kanji_nn.data import tap, plot_mcp
from kanji_nn.cli import *


cpd_channels = ["raw:stness:loc", "gauss:K"]

composed_metrics = compose(
    # tap(partial(plot_mcp, channels=plot_channels, show=False)),
    metrics.local_straightness,
    partial(metrics.tangential_acc, speed_key="raw:speed:central"),
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
