import numpy as np


def cpd_signal(stroke, channels, weights):
    z_score = lambda c: (c - np.mean(c)) / (np.std(c) + 1e-6)
    channels_norm = [z_score(stroke.features[key]) for key in channels]

    # weigthed sum of normalized input channels:
    S = np.dot(np.array(weights), np.array(channels_norm))
    return stroke.clone(features={"S": S})
