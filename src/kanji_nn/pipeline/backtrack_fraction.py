import numpy as np

def backtrack_fraction(stroke, span=14, window_length=7):
    """
    Per-sample backtrack fraction: how much of the local motion (inner
    window) moves opposite to the broader direction of travel (outer span).

    xy: (n, 2) uniformly-resampled polyline.
    span: outer window (samples) used to establish reference direction.
    window_length: inner window (samples) whose steps get projected/scored.

    Returns: (n,) array, NaN where a full outer span isn't available
    (edges) or where chord/step magnitudes are degenerate.
    0 = purely forward motion, -> 1 = dominated by backward egenerate.
    """

    xy = stroke.xy

    n = len(xy)
    half_span = span // 2
    half_win = window_length // 2
    result = np.full(n, np.nan)

    for i in range(half_span, n - half_span):
        p_start, p_end = xy[i - half_span], xy[i + half_span]
        chord = p_end - p_start
        chord_len = np.linalg.norm(chord)
        if chord_len < 1e-9:
            continue
        r_hat = chord / chord_len

        w0, w1 = max(i - half_win, 0), min(i + half_win, n - 1)
        steps = np.diff(xy[w0:w1 + 1], axis=0)
        if len(steps) == 0:
            continue

        proj = steps @ r_hat
        total = np.sum(np.abs(proj))
        if total < 1e-12:
            continue

        result[i] = -np.sum(proj[proj < 0]) / total

    mask = np.where(result > 0)[0]
    if len(mask):
        print(f"{stroke.literal}/{stroke.stroke_index}", len(mask))

    return stroke.clone(features={"backtrack_fraction": result})