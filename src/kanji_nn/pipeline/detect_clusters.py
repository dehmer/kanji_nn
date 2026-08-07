import numpy as np
import math
from functools import partial
import scipy.signal as signal
from kanji_nn.predef import clamp


def find_legs(stroke, peaksfn):
    """
    Pair-up angle peeks and speed valleys. Multiple peaks may hit
    a single valley.
    Return dictionary with valley boundaries as keys and
    list of peak indices as values.
    """
    speed = stroke.features["raw:speed:central"]

    # Find prominent angle peak within distance samples.
    angle_peaks, _ = peaksfn()

    window, k = 10, 2
    def step(x, d, mfn):
        a, b = (max(0, x-window), x) if d < 0 else (x, min(stroke.n_points, x+window))
        m = mfn(slice(a, b))
        if not np.any(m):
            return x, True
        return (a + np.argmax(m), False) if d < 0 else (b - np.argmax(m[::-1]), False)

    def expand(i, j, mfn):
        i, li = step(i, -1, mfn)
        j, rj = step(j, +1, mfn)
        return (i, j) if li and rj else expand(i, j, mfn)

    leg_bounds = {}
    if len(angle_peaks):
        epsilon = np.max(speed[angle_peaks])
        epsilon = clamp(epsilon, 2e-9, epsilon)
        mfn = lambda s: speed[s] < epsilon * k

        for idx in angle_peaks:
            bounds = expand(idx - 1, idx + 1, mfn)
            leg_bounds.setdefault(bounds, []).append(idx)

    return leg_bounds


def estimate_leg_heading(stroke, start, end):
    """
    Fit a directed heading vector to raw xy over [start, end) via
    total-least-squares (PCA principal axis), oriented by net displacement.
    """
    xy = stroke.xy[start:end]
    centered = xy - xy.mean(axis=0)

    # Principal axis = right singular vector for the largest singular value.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    heading = vt[0]

    # PCA axis has a 180° sign ambiguity - orient via net displacement.
    net = xy[-1] - xy[0]
    if np.dot(heading, net) < 0:
        heading = -heading

    return heading


def intersect_lines(p1, d1, p2, d2):
    """
    Intersection of two lines given as point + direction (2D).
    Returns None if lines are (near-)parallel — caller decides fallback.
    """
    a = np.column_stack([d1, -d2])
    det = np.linalg.det(a)
    if abs(det) < 1e-9:
        return None
    t = np.linalg.solve(a, p2 - p1)
    return p1 + t[0] * d1


def compute_leg_agreement(stroke, leg_bounds, leg_window=16):
    """
    For each (i_in, i_out) candidate, estimate clean entry/exit headings
    from windows just outside the contaminated span and compare them.
    Returns signed angle in radians: ~0 = ordinary corner (agreement),
    ~±pi = traceback (opposition).
    """
    n = stroke.n_points
    agreement = {}

    for (i_in, i_out), peaks in leg_bounds.items():
        entry_start = max(0, i_in - leg_window)
        exit_end = min(n, i_out + leg_window)

        entry = estimate_leg_heading(stroke, entry_start, i_in)
        exit_ = estimate_leg_heading(stroke, i_out, exit_end)

        cross = entry[0] * exit_[1] - entry[1] * exit_[0]
        dot = np.dot(entry, exit_)
        signed_angle = math.atan2(cross, dot)

        # Anchor each line at the last/first clean boundary sample.
        p_in = stroke.xy[i_in]
        p_out = stroke.xy[i_out]
        designated_apex = intersect_lines(p_in, entry, p_out, exit_)

        agreement[(i_in, i_out)] = {
            "peaks": peaks,
            "entry_heading": entry,
            "exit_heading": exit_,
            "signed_angle": signed_angle,
            "signed_angle_deg": math.degrees(signed_angle),
            "designated_apex": designated_apex
        }

    return agreement


def plot_props(tracebacks):
    """
    Prepare apices and bounds vlines for plot.
    """

    i_in, i_out, apices = [], [], []
    for bounds, props in tracebacks.items():
        i_in.append(bounds[0])
        i_out.append(bounds[1])
        apices.append(props["designated_apex"])

    apices_options = {"marker": "o", "color": 'orange', "zorder": 3, "alpha": 0.7}
    i_in_options = [{"color": "green", "linestyle": "-.", "linewidth": 1, "alpha": 1.0}] * len(i_in)
    i_out_options = [{"color": "blue", "linestyle": "-.", "linewidth": 1, "alpha": 1.0}] * len(i_out)

    vl_start = list(zip(i_in, i_in_options))
    vl_end = list(zip(i_out, i_out_options))
    vlines = vl_start + vl_end

    apices = np.hstack(apices).reshape(-1, 2)
    scatter = (apices, apices_options)

    return {"vlines": vlines, "scatter": scatter}


def detect_clusters(stroke, distance=1, prominence=math.pi/3):
    """
    Detect vertex clusters in slow regions, i.e. low det displacement
    with relative high turn angles.
    """

    # Find prominent angle peak within distance samples.
    angle = stroke.features["angle:w=1:abs"]
    peaksfn = lambda: signal.find_peaks(angle, distance=distance, prominence=prominence)

    leg_bounds = find_legs(stroke, peaksfn)
    if not leg_bounds:
        return stroke

    # Note:
    # With relative low angle peak prominence (<< pi) indentified bounds are
    # just candidates for possible cleanup of some sort, not actual
    # direction reversals.
    #
    clusters = compute_leg_agreement(stroke, leg_bounds)
    props = plot_props(clusters) | {"clusters": clusters}
    return stroke.clone(props=props)
