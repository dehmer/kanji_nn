import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from svg.path import CubicBezier
from kanji_nn.svg.ScaledPath import ScaledPath


def dtw(stroke):
    """
    """
    xy = stroke.features["gauss:xy"]
    parsed_path = stroke.sticky["path"]

    scaled_path = ScaledPath(parsed_path, error=1e-5)
    path_vertices, path_segments = scaled_path.interpolate_uniform(len(xy))
    distance, path = fastdtw(xy, path_vertices, radius=1, dist=euclidean)

    warp = np.asarray(path)
    segs = path_segments[warp[:, 1]].astype(np.int64)
    D = np.column_stack([warp, segs])
    boundary_rows = np.flatnonzero(np.diff(D[:, 2])) + 1

    # useful to plot segment boundaries:
    indices = D[boundary_rows, 0]
    vline_options = [{
        "color": "gray",
        "linestyle": "-.",
        "linewidth": 1,
        "alpha": 1.0,
    }] * len(indices)

    vlines = list(zip(indices, vline_options))

    # prepare struts columns (for plotting): x1, y1, x2, y2:
    a = xy[D[boundary_rows, 0]]
    b = path_vertices[D[boundary_rows, 1]]
    struts = np.column_stack([a[:, 0], b[:, 0], a[:, 1], b[:, 1]])

    # record same query index addressing consecutive segments
    same_index = D[:-1, 0] == D[1:, 0]
    rising = D[1:, 2] > D[:-1, 2]
    not_moveto = D[:-1, 2] != 0
    conflict_rows = np.flatnonzero(not_moveto & same_index & rising)

    return stroke.clone(props={
        "vlines": vlines,
        "struts": struts,
        "D": D,
        "conflict_rows": conflict_rows
    })
