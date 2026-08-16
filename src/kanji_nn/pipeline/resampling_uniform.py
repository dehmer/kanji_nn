import numpy as np
from scipy.ndimage import gaussian_filter1d
from kanji_nn.data.stroke import Stroke


def resampling_uniform(stroke, n_out=None):
    n_out = stroke.n_points if not n_out else n_out
    n_out = max(n_out, 2)

    s = stroke.features['raw:s']
    samples = np.linspace(0.0, s[-1], n_out)
    x = np.interp(samples, s, stroke.x)
    y = np.interp(samples, s, stroke.y)
    xy = np.column_stack([x, y])

    return stroke.clone(props={"resampled:xy", xy})
