import numpy as np


def density():
    ds = []
    def inner(stroke):
        nonlocal ds
        s = stroke.features["raw:s"]
        arc_length = s[-1]
        ds.append(arc_length / stroke.n_points)

        dsa = np.asarray(ds)
        print("ds:", "\tmin", np.min(dsa), "\tmax", np.max(dsa), "\tmean", np.mean(dsa), "\tstd", np.std(dsa))

        return stroke
    return inner