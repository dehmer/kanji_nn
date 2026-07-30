import numpy as np
from svg.path import CubicBezier
from kanji_nn.svg.bezier_obb import bezier_obb
from kanji_nn.svg.classify_bezier import classify_bezier


def resolve_segments(stroke):
    xy = stroke.features["gauss:xy"]
    D = stroke.props["D"]
    parsed_path = stroke.sticky["path"]

    # currently unused:
    conflict_rows = stroke.props["conflict_rows"]
    obb_ratios = [None] + [bezier_obb(s)["ratio"] for s in parsed_path if isinstance(s, CubicBezier)]
    bezier_classes = ["move-to"] + [classify_bezier(s) for s in parsed_path if isinstance(s, CubicBezier)]

    # 3-bucket rule:
    # 1. NS/NS => first segment (near-straight)
    # 2. S/NS, C/NS => curvy segment
    # 3. C/C => keep both

    # Pull xy for each segment and attach parsed path:
    # Keep conflicting pairs (i, s), (i, s+n), n > 0 as is.
    # No 3-bucket rule for now.
    splits = np.where(np.diff(D[:, 2]) != 0)[0] + 1
    segments = [
        # [start, end)
        (xy[(s[0, 0]):(s[-1, 0] + 1), :], parsed_path[s[0, 1]])
        for s in np.split(D[:, (0, 2)], splits)
    ]

    return stroke.clone(props={"segments": segments})
