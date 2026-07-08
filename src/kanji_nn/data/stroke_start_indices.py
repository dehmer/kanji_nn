import numpy as np

def stroke_start_indices(tensor):
    """Row indices where a new stroke begins."""
    PEN = 3
    pen = tensor[:, PEN]
    eos = np.where(pen == 0)[0]
    starts = eos + 1
    return np.concatenate(([0], starts[starts < len(tensor)]))
