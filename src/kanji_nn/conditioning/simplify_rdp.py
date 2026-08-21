import numpy as np
from rdp import rdp
from kanji_nn.data.stroke import Stroke

def simplify_rdp(stroke, epsilon=1e-4):
    """
    epsilon: the smaller the more points
    """
    xy = stroke.features["xy"]
    z = np.zeros(len(xy))
    xyz = np.column_stack((xy, z))
    simplified = rdp(xyz, epsilon=epsilon)[:, :-1]

    return Stroke(
        dataset=stroke.dataset,
        key=stroke.key,
        sticky=stroke.sticky,
        features={"xy": simplified}
    )
