import numpy as np
from svg.path import CubicBezier
from kanji_nn.data.bezier_obb import bezier_obb
from kanji_nn.data.classify_bezier import classify_bezier


def resolve_segments(stroke):

    D = stroke.props["D"]
    conflict_rows = stroke.props["conflict_rows"]
    parsed_path = stroke.sticky["path"]
    obb_ratios = [None] + [bezier_obb(s)["ratio"] for s in parsed_path if isinstance(s, CubicBezier)]
    bezier_classes = ["move-to"] + [classify_bezier(s) for s in parsed_path if isinstance(s, CubicBezier)]

    # print(obb_ratios)
    # print(bezier_classes)
    # for c in bezier_classes:
    #     print(f"{stroke.literal}/{stroke.stroke_index}", c)

    for i in conflict_rows:
        print(
            f"{stroke.literal}/{stroke.stroke_index} - segment conflict @ {D[i, 0]}:",
            f"{D[i, 2]}: {bezier_classes[D[i, 2]]} ({obb_ratios[D[i, 2]]:.3f})",
            "->",
            f"{D[i + 1, 2]}: {bezier_classes[D[i + 1, 2]]} ({obb_ratios[D[i + 1, 2]]:.3f})"
        )

    # TODO: ...

    return stroke
