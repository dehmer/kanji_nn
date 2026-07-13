import numpy as np

def local_straightness(stroke):
    # Real local straightness: Ratio of straight-line shortcut to actual path arc
    # Window size of 3 points (i-1 to i+1)
    chord = np.hypot(stroke.x[2:] - stroke.x[:-2], stroke.y[2:] - stroke.y[:-2])
    s = stroke.features['s']
    arc = s[2:] - s[:-2]

    # Pad ends to keep matching array shapes
    local_straightness = chord / np.where(arc == 0, 1e-6, arc)
    metric = np.pad(local_straightness, (1, 1), mode='edge')

    return stroke.clone(features={"local_straightness": metric})
