import numpy as np
import math

def detect_tracebacks(stroke, speed_epsilon=1e-3, reversal_frac=0.4):
    tracebacks = stroke.props.get("tracebacks", [])

    angle = stroke.features["angle:w=1:abs"]
    speed = stroke.features["raw:speed:central"]
    txy = stroke.features["gauss:txy"]

    mask = (angle > math.pi * reversal_frac) & (speed < speed_epsilon)
    indices = np.where(mask)[0]

    if len(indices):
        print(f"{stroke.literal}/{stroke.stroke_index}: ", indices)

    # TODO: compare incoming/outgoing legs, e.g. reverse tangent agreement
    # TODO: store apex, p_in, p_out in stroke props
    # TODO: if traceback was detected call detect_tracebacks again and merge

    vline_options = [{
        "color": "gray",
        "linestyle": "-.",
        "linewidth": 1,
        "alpha": 1.0,
    }] * len(indices)

    vlines = list(zip(indices, vline_options))

    return stroke.clone(props={"vlines": vlines})
