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


def measure_traceback(stroke, peak_idx, window=5):
    window = window + 1

    # Don't event bother if window extends over data bounds.
    if peak_idx - window < 0 or peak_idx + window >= stroke.n_points:
        return np.full(window, np.nan)

    xy = stroke.xy
    apex = xy[peak_idx]

    dots = np.array([
        np.dot(
            xy[peak_idx + k] - apex,
            xy[peak_idx - k] - apex
        )
        for k in range(1, window)
    ])

    # TODO: zero-norm gap might bite us below.
    norms = np.array([
        np.linalg.norm(xy[peak_idx + k] - apex) *
        np.linalg.norm(xy[peak_idx - k] - apex)
        for k in range(1, window)
    ])

    return dots / norms


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

        v_entry = estimate_leg_heading(stroke, entry_start, i_in)
        v_exit = estimate_leg_heading(stroke, i_out, exit_end)
        cross = v_entry[0] * v_exit[1] - v_entry[1] * v_exit[0]
        dot = np.dot(v_entry, v_exit)
        signed_angle = math.atan2(cross, dot)

        # Anchor each line at the last/first clean boundary sample.
        p_in = stroke.xy[i_in]
        p_out = stroke.xy[i_out]
        designated_apex = intersect_lines(p_in, v_entry, p_out, v_exit)

        for peak_idx in peaks:
            traceback = measure_traceback(stroke, peak_idx)
            if traceback[0] == 1.0:
                print(f"{stroke.literal}/{stroke.stroke_index} - peak_idx={peak_idx}, traceback={traceback}")

        agreement[(i_in, i_out)] = {
            "peaks": peaks,
            "entry_heading": v_entry,
            "exit_heading": v_exit,
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
    else:
        # print(f"{stroke.literal}/{stroke.stroke_index} -", leg_bounds)
        pass

    # Note: With relative low angle peak prominence (<< pi) indentified
    # bounds are just candidates for possible cleanup of some sort,
    # not necessarily direction reversals only.
    #
    clusters = compute_leg_agreement(stroke, leg_bounds)
    # print(clusters)

    # For single cluster, zoom-in on designated apex:
    zoom = None
    if len(clusters) == 1:
        apex = list(clusters.values())[0]["designated_apex"]
        xlim = (clamp(apex[0] - 0.15, 0.0, 1.0), clamp(apex[0] + 0.15, 0.0, 1.0))
        ylim = (clamp(apex[1] - 0.15, 0.0, 1.0), clamp(apex[1] + 0.15, 0.0, 1.0))
        zoom = (xlim, ylim)

    props = plot_props(clusters) | {"clusters": clusters, "zoom": zoom}
    # props = {"clusters": clusters, "zoom": zoom}
    return stroke.clone(props=props)
