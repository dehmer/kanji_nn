import numpy as np

def pressure(stroke):
    pressure = stroke.pressure

    # fancy way to say pn = p / p.max() if p.min() = 0 (which it actually is):
    pressure_norm = (pressure - pressure.min()) / (pressure.max() - pressure.min())
    return stroke.clone(features={"P": pressure, "P_norm": pressure_norm})
