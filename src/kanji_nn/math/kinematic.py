import numpy as np

def kinematic(data):
    dt = np.concatenate(([0.0], np.diff(data['timestamp'])))
    dx = np.concatenate(([0.0], np.diff(data['x'])))
    dy = np.concatenate(([0.0], np.diff(data['y'])))

    distance = np.sqrt(dx**2 + dy**2)
    vector = np.column_stack((dx, dy))
    magnitude = np.linalg.norm(vector, axis=1, keepdims=True)
    magnitude = np.where(magnitude == 0, 1.0, magnitude)
    unit_vector = vector / magnitude

    dot_product = np.sum(unit_vector[:-1] * unit_vector[1:], axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    velocity = distance / dt * 1000

    # θ (theta)
    angle = np.degrees(np.arccos(dot_product))
    angle = np.pad(angle, (0, 1), mode='constant', constant_values=np.nan)

    return {
        'velocity': velocity,
        'angle': angle
    }
