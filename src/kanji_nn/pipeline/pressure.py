import numpy as np

def pressure(stroke):
    p = stroke.pressure

    # fancy way to say pn = p / p.max() if p.min() = 0 (which it actually is):
    p_norm = (p - p.min()) / (p.max() - p.min())
    p_inv = (1.0 - p_norm)
    dp = np.diff(p, prepend=0)

    return stroke.clone(features={"P": p, "P:norm": p_norm, "P:inv": p_inv, "dP": dp})
