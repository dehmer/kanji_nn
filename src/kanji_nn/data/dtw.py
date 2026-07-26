import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from svg.path import CubicBezier
from .ScaledPath import ScaledPath
from kanji_nn.data.bezier_obb import bezier_obb


def dtw(stroke):
    error = 1e-5
    xy = stroke.features["gauss:xy"]
    parsed_path = stroke.sticky["path"]
    obb_ratios = [None] + [bezier_obb(s)["ratio"] for s in parsed_path if isinstance(s, CubicBezier)]

    scaled_path = ScaledPath(parsed_path, error=error)
    path_vertices, path_segments = scaled_path.interpolate_uniform(len(xy))
    distance, path = fastdtw(xy, path_vertices, radius=1, dist=euclidean)

    warp = np.asarray(path)
    segs = path_segments[warp[:, 1]].astype(np.int64)
    D = np.column_stack([warp, segs])
    boundary_rows = np.flatnonzero(np.diff(D[:, 2])) + 1

    same_index = D[:-1, 0] == D[1:, 0]
    rising = D[1:, 2] > D[:-1, 2]
    not_moveto = D[:-1, 0] != 0
    conflict_rows = np.flatnonzero(not_moveto & same_index & rising)

    for i in conflict_rows:
        print(
            f"{stroke.literal}/{stroke.stroke_index} - segment conflict [{D[i, 0]}]:",
            D[i, 2], "->", D[i + 1, 2],
            "[S]", obb_ratios[D[i, 2]], obb_ratios[D[i + 1, 2]]
        )

    indices = D[boundary_rows, 0]
    vline_options = [{
        "color": "gray",
        "linestyle": "-.",
        "linewidth": 1,
        "alpha": 1.0,
    }] * len(indices)

    vlines = list(zip(indices, vline_options))

    # prepare struts columns: x1, y1, x2, y2
    a = xy[D[boundary_rows, 0]]
    b = path_vertices[D[boundary_rows, 1]]
    struts = np.column_stack([a[:, 0], b[:, 0], a[:, 1], b[:, 1]])

    return stroke.clone(props={"vlines": vlines, "struts": struts})
