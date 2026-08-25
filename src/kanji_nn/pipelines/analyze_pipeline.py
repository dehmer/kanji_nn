from functools import partial

from kanji_nn.data import Character
from kanji_nn.predef import compose, tap
import kanji_nn.analysis as analysis
import kanji_nn.plot as plot
import kanji_nn.metrics as metrics
import kanji_nn.conditioning as conditioning


plot_channels=[
    "pressure",
    "dP/dt",
    "raw:speed:backward",
    "raw:speed:central",
    "gauss:dθ/ds:abs",
    "raw:stness:loc",
    "raw:speed:forward",
    "raw:stness",
    # "angle:w=1:abs",
    # "raw:speed:central",
]


def analyze_pipeline():
    return compose(
        tap(partial(plot.show_mcp_plot, show=True, save=False, channels=plot_channels)),
        # analysis.density(),
        # metrics.arc_length_raw,
        metrics.local_straightness,
        metrics.straightness,
        metrics.curvature,
        metrics.tangent,
        metrics.arc_length_gauss,
        metrics.arc_length_raw,
        metrics.central_speed,
        metrics.backward_speed,
        metrics.forward_speed,
        metrics.pressure_derivative,
        conditioning.split_raw,
        tap(lambda s: print(f"{s.dataset} - {s.literal}/{s.stroke_index}"))
    )
