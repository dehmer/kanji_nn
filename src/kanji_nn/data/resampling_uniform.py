import numpy as np
from scipy.ndimage import gaussian_filter1d
from kanji_nn.plot import strokes_plot

def resampling_uniform(stroke, sigma=1.0, mode="reflect"):

    s = stroke.features['raw:s']
    samples = np.linspace(0.0, s[-1], stroke.n_points)

    # keep x/y separate for now:
    x = np.interp(samples, s, stroke.x)
    y = np.interp(samples, s, stroke.y)
    xy = np.column_stack([x, y])

    xy_gauss = gaussian_filter1d(xy, axis=0, sigma=sigma, mode=mode)
    title = f"Resampled, gaussian filter\nσ = {sigma}, mode = {mode}"
    strokes_plot.show([xy_gauss], title=title, alpha=0.1)


    return xy_gauss
