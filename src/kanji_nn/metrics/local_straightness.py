import numpy as np

def local_straightness(stroke):
    # Real local stness: Ratio of straight-line shortcut to actual path arc
    # Window size of 3 points (i-1 to i+1)
    chord = np.hypot(stroke.x[2:] - stroke.x[:-2], stroke.y[2:] - stroke.y[:-2])
    s = stroke.features['s']
    arc = s[2:] - s[:-2]

    # Pad ends to keep matching array shapes
    loc_stness = chord / np.where(arc == 0, 1e-6, arc)
    metric = np.pad(loc_stness, (1, 1), mode='edge')

    return stroke.clone(features={"loc_stness": metric})
