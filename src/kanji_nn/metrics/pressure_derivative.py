import numpy as np

def pressure_derivative(stroke):
    pressure = stroke.features["pressure"]
    t = stroke.features["t"]
    dP_dt = np.gradient(pressure, t)
    return stroke.clone(features={"dP/dt": dP_dt})
