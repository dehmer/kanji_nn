import numpy as np
from kanji_nn.math import calc_kinematic

def dehook(stroke, window=(150, 50), velocity_threshold=0.5, angle_threshold=60):
    blocks = {'raw': stroke}
    blocks = blocks | calc_kinematic(blocks)

    timestamp = blocks['raw'][:, 0]
    timestamp = timestamp - timestamp[0]
    timestamp_start = timestamp[0]
    timestamp_end = timestamp[-1]

    # adjust window for lines shorter than window.
    if (window[0] + window[1]) > timestamp_end:
        return stroke

    velocity = blocks['velocity']
    angle = blocks['angle']
    condition = (velocity < velocity_threshold) & (angle >= angle_threshold)
    candidates = np.where(condition)[0]

    # FIXME: might not provide expected result
    left_bound = np.where(timestamp >= window[0])[0][0]
    right_bound = np.where(timestamp >= timestamp_end - window[1])[0][0]

    start, end = 0, len(stroke)
    for candidate in candidates:
        if timestamp[candidate] <= timestamp[left_bound]:
            start = candidate + 1
        if timestamp[candidate] >= timestamp[right_bound]:
            end = candidate - 1

    return stroke[start:end,:]
