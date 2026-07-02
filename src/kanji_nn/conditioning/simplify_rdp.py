import numpy as np
from rdp import rdp

def simplify_rdp(stroke, epsilon=0.01):
    """
    epsilon: the smaller the more points
    """

    xy = stroke[:, [1, 2]]
    print(xy.shape, stroke.shape)
    z = np.zeros((stroke.shape[0], 1))
    stroke = np.hstack((xy, z))
    return rdp(stroke, epsilon=epsilon)[:, :-1]
