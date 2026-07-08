import numpy as np
from .stroke_start_indices import stroke_start_indices

def local_pace(tensor):
    """
    Per-row local pacing: s(t) increment from the previous point within
    the same stroke. Rows at stroke starts (incl. row 0) get NaN — they're
    cross-stroke jump vectors, not paced segments, and get no pacing signal.
    """
    s = tensor[:, 2]
    pace = np.diff(s, prepend=np.nan)
    pace[stroke_start_indices(tensor)] = np.nan
    return pace
