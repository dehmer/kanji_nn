from .determine_knot_continuity import determine_knot_continuity
from .is_regular_path import is_regular_path
from .calculate_knot_angles import calculate_knot_angles


path_fn = lambda s: s.sticky["path"]


def parametric_continuity(stroke, path_fn=path_fn):
    path = path_fn(stroke)
    continuities = determine_knot_continuity(path)
    angles = calculate_knot_angles(path)

    if not is_regular_path(path):
        print(f"{stroke.literal}/{stroke.stroke_index} - irregular path")

    if len(path) > 2:
        print(f"{stroke.literal}/{stroke.stroke_index} -", continuities, angles)

    return stroke
