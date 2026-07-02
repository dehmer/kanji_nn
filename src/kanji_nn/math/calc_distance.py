import numpy as np

def calc_distance(blocks):
    """
    Calculate Euklidean distance of consecutive points.
    Note: We retain distances between strokes.
    """
    blocks = blocks if 'diff' in blocks else blocks | calc_diff(blocks)

    dx = blocks['diff'][:, 1]
    dy = blocks['diff'][:, 2]

    distance = np.sqrt(dx**2 + dy**2)
    return {'distance': distance}
