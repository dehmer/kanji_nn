from functools import partial

from kanji_nn.predef import compose, tap
import kanji_nn.io as io
import kanji_nn.plot as plot
import kanji_nn.conditioning as conditioning
import kanji_nn.metrics as metrics
import kanji_nn.bezier as bezier


def dump():
    first = True
    def inner(stroke):
        nonlocal first
        if first:
            first = False
            print("dataset,stroke,head,tail")

        print(f"{stroke.dataset},{stroke.key},{stroke.props["cuts"][0]},{stroke.props["cuts"][1]}")
    return inner


def compare(cuts_target):
    max = 0.0
    def inner(stroke):
        head = f"{stroke.literal}/{stroke.stroke_index} - "

        if not stroke.key in cuts_target:
            return

        nonlocal max
        a = stroke.props["cuts"]
        e = cuts_target[stroke.key]
        d = (abs(e[0] - a[0]), abs(e[1] - a[1]))
        if d != (0, 0):
            print(head, f"expected: {e}, actual: {a}, difference: {d}")

    return inner


def inject_cuts_target(stroke, cuts_target):
    return stroke.clone(props={"cuts_target": cuts_target[stroke.key]})


plot_channels=[
    "angle:w=1"
]

png_trimmed = lambda s: f"data/dataset/{s.dataset}/png-trimmed/{s.code_point}"


def trim_pipeline(cuts_target):
    ds = 0.006      # slightly below minimum of all strokes
    sigma = 2.0     # Gauss 1D Filter
    return compose(

        # plot.show_strokes_plot(lambda s: s.features["xy"]),
        plot.save_strokes_plot(filename_fn=png_trimmed),
        io.save_npy("npy-trimmed"),
        conditioning.trim_region,
        tap(partial(plot.show_mcp_plot, show=True, save=False, channels=plot_channels)),
        tap(dump()),
        tap(compare(cuts_target)),
        conditioning.dtw_rle,
        partial(bezier.resample_fixed_distance, ds=ds),
        partial(metrics.turning_angle, w=1),

        partial(conditioning.replace_xy, key="gauss:xy"),
        partial(conditioning.gauss_1d, sigma=sigma, f=None),
        partial(conditioning.resample_xy_equidistant, ds=ds),

        conditioning.prune,
        conditioning.split_raw,
        tap(lambda s: print(f"{s.dataset} - {s.literal}/{s.stroke_index}"))
    )
