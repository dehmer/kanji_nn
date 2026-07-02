import numpy as np
from . import calc_diff
from . import calc_distance

def calc_kinematic(blocks):
    blocks = blocks if 'diff' in blocks else blocks | calc_diff(blocks)
    blocks = blocks if 'distance' in blocks else blocks | calc_distance(blocks)

    dt = blocks['diff'][:, 0]
    dx = blocks['diff'][:, 1]
    dy = blocks['diff'][:, 2]
    distance = blocks['distance']

    vector = np.column_stack((dx, dy))
    magnitude = np.linalg.norm(vector, axis=1, keepdims=True)
    magnitude = np.where(magnitude == 0, 1.0, magnitude)
    unit_vector = vector / magnitude

    dot_product = np.sum(unit_vector[:-1] * unit_vector[1:], axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    velocity = distance / dt * 1000
    angle = np.degrees(np.arccos(dot_product))
    angle = np.pad(angle, (0, 1), mode='constant', constant_values=np.nan)

    return blocks | {
        'velocity': velocity,
        'angle': angle
    }
