#!/usr/bin/env python3

import csv
from functools import partial
import numpy as np

import kanji_nn.pipeline as pipeline
from kanji_nn.data import Character, Stroke
from kanji_nn.data import plot_mcp
from kanji_nn.cli import *
from kanji_nn.predef import compose, tap

plot_channels = ["raw:stness:loc", "gauss:K"]

composed_metrics = compose(
    tap(partial(plot_mcp, channels=plot_channels, show=True)),
    pipeline.local_straightness,
    partial(pipeline.tangential_acc, speed_key="raw:speed:central"),
    pipeline.vector_acc,
    pipeline.curvature,
    pipeline.tangent,
    pipeline.central_speed,
    pipeline.straightness,
    pipeline.arc_length,
    pipeline.pressure_derivative,
    pipeline.pressure
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
    strokes = [stroke for stroke in strokes if stroke.sticky["kvg_type"][1] == "HZ"]
    strokes = [composed_metrics(stroke) for stroke in strokes]
