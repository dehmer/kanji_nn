import numpy as np
from rdp import rdp

def simplify_rdp(stroke, epsilon=0.05):
    xy = stroke.xy
    z = np.zeros(len(xy))
    xyz = np.column_stack((xy, z))

    mask = rdp(xyz, epsilon=epsilon, return_mask=True)
    rdp_xy = xy[mask]
    return stroke.clone(props={"rdp:xy": rdp_xy})
