import numpy as np

def tangential_acc(stroke, speed_key):
    """Calculates the rate of change of linear speed."""

    # Choose which speed profile to differentiate
    speed = stroke.features[speed_key]

    # Differentiate speed with respect to time
    return stroke.clone(features={"at": np.gradient(speed, stroke.t)})
