import numpy as np
from .resample_fixed_distance import sample_path


def kvg_bbox(glyph):
    literal = glyph["literal"]
    paths = glyph["kvg:paths"]
    xys = [sample_path(p) for p in paths]
    xys = np.vstack(xys)
    xy = xys[:, :-1]

    mins = np.min(xy, axis=0)
    maxs = np.max(xy, axis=0)

    # [min_x, min_y, max_x, max_y]
    kvg_bbox = np.concat((mins, maxs))

    return glyph | {"kvg:bbox": kvg_bbox}
