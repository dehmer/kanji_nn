import numpy as np
from .stroke_start_indices import stroke_start_indices
from .structural_zero_mask import structural_zero_mask

def random_walk_delta_noise(tensor, sigma, rho=0.85, rng=None):
    rng = rng or np.random.default_rng()
    tensor = tensor.copy()
    n = len(tensor)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float32), (n,))
    start_set = set(stroke_start_indices(tensor).tolist())
    zero_mask = structural_zero_mask(tensor)

    noise = np.zeros((n, 2), dtype=np.float32)
    prev = np.zeros(2)
    for i in range(n):
        prev = rng.normal(0, sigma[i], size=2) if i in start_set else rho * prev + rng.normal(0, sigma[i], size=2)
        noise[i] = prev
    noise[zero_mask] = 0.0

    tensor[:, :2] += noise
    return tensor
