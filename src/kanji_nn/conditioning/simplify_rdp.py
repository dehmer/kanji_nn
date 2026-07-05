import numpy as np
from rdp import rdp

def simplify_rdp(stroke, xy_cols=(0, 1), epsilon=0.005):
    """
    epsilon: the smaller the more points
    """

    z = np.zeros((stroke.shape[0], 1))
    stroke = np.hstack((stroke[:, xy_cols], z))
    return rdp(stroke, epsilon=epsilon)[:, :-1]
