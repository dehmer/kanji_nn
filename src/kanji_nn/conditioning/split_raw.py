import numpy as np

def split_raw(stroke):
    raw = stroke.features["raw"]
    column_count = raw.shape[1]

    # 2 columns: xy only
    # 4 columns: timestamp, xy, pressure
    xy = raw[:, 1:3] if column_count == 4 else raw
    t = raw[:, 0] if column_count == 4 else np.arange(len(xy))
    pressure = raw[:, -1] if column_count == 4 else np.zeros(len(xy))

    return stroke.clone(features={"t": t, "xy": xy, "pressure": pressure})
