from functools import partial

from kanji_nn.predef import compose, tap
import kanji_nn.io as io
import kanji_nn.plot as plot
import kanji_nn.conditioning as conditioning
import kanji_nn.bezier as bezier
import kanji_nn.metrics as metrics


png_raw     = lambda s: f"data/dataset/{s.dataset}/png-raw/{s.code_point}"
png_trimmed = lambda s: f"data/dataset/{s.dataset}/png-trimmed/{s.code_point}"
png_fitted  = lambda s: f"data/dataset/{s.dataset}/png-fitted/{s.code_point}"
npy_fitted  = lambda s: f"data/dataset/{s.dataset}/npy-fitted/{s.code_point}.npy"
fitted_path = lambda s: s.props["fitted"]
xys         = lambda s: s.props["path:xys"][:, :-1]


def post_pipeline():
    ds = 0.006      # slightly below minimum of all strokes
    alpha = 0.1
    sigma = 2.0     # Gauss 1D Filter
    epsilon = 3e-4  # RDP
    maxError = 5e-4 # Schneider's Algorithm

    return compose(
        io.save_splines(filename_fn=npy_fitted),
        partial(plot.save_strokes_plot(filename_fn=png_fitted, xy_fn=xys, alpha=alpha)),
        partial(bezier.resample_fixed_distance, path_fn=fitted_path, ds=ds),
        partial(bezier.schneider, maxError=maxError),
        partial(conditioning.simplify_rdp, epsilon=epsilon),

        partial(plot.save_strokes_plot(filename_fn=png_trimmed, alpha=alpha)),
        conditioning.trim_region,
        conditioning.dtw_rle,
        partial(metrics.turning_angle, w=1),
        partial(bezier.resample_fixed_distance, ds=ds),

        # Note: gauss_1d works best for uniformly sampled point w.r.t to arc-length spacing.
        # Hence resample_xy_equidistant first with ds=0.006.
        partial(conditioning.replace_xy, key="gauss:xy"),
        partial(conditioning.gauss_1d, sigma=sigma, f=None),
        partial(conditioning.resample_xy_equidistant, ds=ds),

        conditioning.prune,
        partial(plot.save_strokes_plot(filename_fn=png_raw, alpha=alpha)),
        conditioning.split_raw,
        tap(lambda s: print(f"{s.dataset} - {s.literal}/{s.stroke_index}")),
    )
