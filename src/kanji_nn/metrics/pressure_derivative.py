import numpy as np

def pressure_derivative(stroke):
    dP_dt = np.gradient(stroke.pressure, stroke.t)
    return stroke.clone(features={"dP/dt": dP_dt})
