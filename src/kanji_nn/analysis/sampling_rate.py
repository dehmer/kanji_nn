import numpy as np


def sampling_rate(stroke):
    dt = np.diff(stroke.t)
    min, max, mean, std = np.min(dt), np.max(dt), np.mean(dt), np.std(dt)
    print(f"{min = }")
    print(f"{max = }")
    print(f"{mean = }")
    print(f"{std = }")
    return stroke
