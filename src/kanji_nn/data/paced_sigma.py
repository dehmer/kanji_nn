import numpy as np
from .local_pace import local_pace

def paced_sigma(tensor, sigma_base, k, weight_range=(0.2, 5.0)):
    """
    k > 0: more noise on large-pace (fast, straight) segments.
    k < 0: more noise on small-pace (slow, careful/curved) segments.
    k = 0: uniform, falls back to plain random_walk_delta_noise.
    """
    pace = local_pace(tensor)
    default = np.nanmedian(pace) if np.any(~np.isnan(pace)) else 1.0
    pace = np.where(np.isnan(pace), default, pace)
    pace = np.clip(pace, 1e-4, None)          # guard against 0**negative_k
    weight = np.clip(pace ** k, *weight_range) # guard against blowup either direction
    return sigma_base * weight
