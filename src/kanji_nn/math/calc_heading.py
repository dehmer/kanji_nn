import numpy as np

def calc_heading(blocks):
    dx = blocks['diff'][:, 1]
    dy = blocks['diff'][:, 2]
    heading = np.arctan2(dy, dx)
    return {'heading': heading}
