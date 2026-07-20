#!/usr/bin/env python3

import csv
from functools import partial
import numpy as np

import kanji_nn.metrics as metrics
from kanji_nn.data import Character, Stroke, compose
from kanji_nn.data import find_trim_region, tap
from kanji_nn.cli import *


cpd_channels = ["raw:stness:loc", "gauss:K"]

composed_metrics = compose(
    find_trim_region,
    partial(metrics.cpd_signal, channels=cpd_channels),
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
    # strokes = [stroke for stroke in strokes if stroke.literal == '石']
    strokes = sorted(strokes, key = lambda s: s.literal)

    for stroke in strokes:
        expected = stroke.props["cuts"]
        stroke = composed_metrics(stroke)
        actual = stroke.props["cuts"]
        error = (actual[0] - expected[0], actual[1] - expected[1])
        print(stroke.literal, stroke.stroke_index, stroke.stroke_type, f"expected={expected}\tactual={actual}\terror={error}")
