from functools import partial

from kanji_nn.data import Character
from kanji_nn.predef import compose, tap
import kanji_nn.bezier as bezier
import kanji_nn.io as io
import kanji_nn.plot as plot
import kanji_nn.conditioning as conditioning
import kanji_nn.bezier as bezier


fitted_path = lambda s: s.props["fitted"]
xys         = lambda s: s.props["path:xys"][:, :-1]
png_fitted  = lambda s: f"data/dataset/{s.dataset}/png-fitted/{s.code_point}"
npy_fitted  = lambda s: f"data/dataset/{s.dataset}/npy-fitted/{s.code_point}.npy"


def fit_pipeline():
    ds = 0.006      # slightly below minimum of all strokes
    alpha = 0.1
    epsilon = 3e-4  # RDP
    maxError = 5e-4 # Schneider's Algorithm

    return compose(
        io.save_splines(filename_fn=npy_fitted),
        partial(plot.save_strokes_plot(filename_fn=png_fitted, xy_fn=xys, alpha=alpha)),
        partial(bezier.resample_fixed_distance, path_fn=fitted_path, ds=ds),
        partial(bezier.schneider, maxError=maxError),

        partial(conditioning.simplify_rdp, epsilon=epsilon),
        conditioning.split_raw,
        tap(lambda s: print(f"{s.dataset} - {s.literal}/{s.stroke_index}"))
    )
