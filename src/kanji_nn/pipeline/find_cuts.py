import numpy as np
import numpy.polynomial.polynomial as polynomial
import math
from scipy.signal import argrelextrema, find_peaks
import scipy.signal as signal

def find_cuts(stroke):
    t = stroke.t - stroke.t[0]
    angle = stroke.features["angle:w=1:abs"]
    speed = stroke.features["raw:speed:central"].copy()



    return stroke
