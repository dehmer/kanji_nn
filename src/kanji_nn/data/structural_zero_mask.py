import numpy as np

def structural_zero_mask(tensor):
    """Rows whose (Δx, Δy) are zero by definition, not by geometry."""
    mask = np.zeros(len(tensor), dtype=bool)
    mask[0] = True
    PEN = 3
    mask[tensor[:, PEN] == 0] = True
    return mask
