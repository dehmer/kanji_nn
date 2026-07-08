import numpy as np
from .local_pace import local_pace

def paced_sigma(tensor, sigma_base, k, weight_range=(0.2, 5.0)):
    """
    weight_range — the hard ceiling/floor on the pace-based multiplier,
    independent of k's magnitude. This is your safety rail: even with a
    large |k|, weight_range caps how far sigma can swing in the most
    extreme (densest/sparsest) regions. If image 3's loop sits exactly
    in a dense-curve region, that's consistent with k negative pushing
    sigma toward the range's ceiling right where the point density (and
    hence duration of sustained correlated drift) is also highest — a
    double whammy in the same spot.
    """
    pace = local_pace(tensor)
    default = np.nanmedian(pace) if np.any(~np.isnan(pace)) else 1.0
    pace = np.where(np.isnan(pace), default, pace)
    pace = np.clip(pace, 1e-4, None)          # guard against 0**negative_k
    weight = np.clip(pace ** k, *weight_range) # guard against blowup either direction
    return sigma_base * weight
