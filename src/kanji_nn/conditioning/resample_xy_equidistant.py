import numpy as np
from kanji_nn.data.stroke import Stroke


def resample_xy_equidistant(stroke, ds=0.006):
    """
    Resample a polyline (n, 2) to uniform arc-length spacing `ds`.
    Note: ds value of 0.006 is slightly below minimum of all strokes in
    current corpus.

    Uses linear interpolation between original samples -- appropriate
    for raw pen data, not a curve reconstruction. The last output
    point is always the original path's true endpoint (final segment
    may be shorter than ds).

    Returns: (m, 2) array of resampled points.
    """
    xy = stroke.features["xy"]
    deltas = np.diff(xy, axis=0)
    step_lengths = np.sqrt(np.sum(deltas**2, axis=1))
    cum_dist = np.concatenate(([0.0], np.cumsum(step_lengths)))
    total_length = cum_dist[-1]

    if total_length < 1e-9:
        return xy[:1].copy()

    n_full = int(total_length // ds)
    target_dist = np.arange(n_full + 1) * ds
    if target_dist[-1] < total_length - 1e-9:
        target_dist = np.concatenate([target_dist, [total_length]])

    x = np.interp(target_dist, cum_dist, xy[:, 0])
    y = np.interp(target_dist, cum_dist, xy[:, 1])
    xy = np.column_stack([x, y])

    return Stroke(
        dataset=stroke.dataset,
        key=stroke.key,
        sticky=stroke.sticky,
        features={"xy": xy}
    )
