import numpy as np
from .resample_fixed_distance import sample_path
from .spline_array import paths_to_array
from .splines_bbox import splines_bbox


def kvg_bbox(glyph):
    paths = glyph["kvg:paths"]
    splines = paths_to_array(paths)
    return glyph | {"kvg:bbox": splines_bbox(splines)}
