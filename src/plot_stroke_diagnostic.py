#!/usr/bin/env python3

"""
Stroke diagnostic plot: colors a single stroke's path by an arbitrary
per-sample channel (pressure, speed, curvature, ...) so that geometric
artifacts (hooks, loops, backswings) can be visually correlated against
that channel at a glance.

Two synchronized panels:
  left  - the (x, y) path, drawn as a color-coded line (LineCollection),
          with start (green) / end (blue) markers matching your existing
          convention, and every raw sample plotted as a small dot.
  right - the channel value against sample index (or cumulative arc
          length), with the same colormap/normalization, plus a
          horizontal reference line (e.g. the pressure floor) if given.

Because both panels share the same color normalization, you can find a
dip/spike in the right panel and immediately see *where on the path* it
happened in the left panel, and vice versa.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def _segments(x, y):
    """Build line segments for LineCollection: one per consecutive pair."""
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def plot_stroke_diagnostic(
    x, y, channel, *,
    t=None,
    channel_label="channel",
    cmap="viridis",
    ref_line=None,
    ref_label=None,
    title=None,
    invert_y=True,
    figsize=(11, 4.5),
):
    """
    Parameters
    ----------
    x, y : array-like, same length
        Stroke coordinates (already in whatever normalized space you use).
    channel : array-like, same length as x/y
        Per-sample value to color by, e.g. pressure, local speed,
        curvature, orientation.
    t : array-like, optional
        Timestamps (ms). If given, right panel x-axis is time; otherwise
        sample index is used.
    ref_line : float, optional
        Draws a horizontal reference line at this value on the right
        panel (e.g. the pressure floor 1/3 raw -> 0.08 normalized).
    invert_y : bool
        Flip y-axis so it matches typical screen coordinates
        (origin top-left). Set False if your y is already "math" oriented.

    Returns
    -------
    fig, (ax_path, ax_channel)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    channel = np.asarray(channel, dtype=float)
    n = len(x)
    assert len(y) == n and len(channel) == n, "x, y, channel must be same length"

    idx = np.arange(n)
    xvals = np.asarray(t, dtype=float) if t is not None else idx

    norm = Normalize(vmin=np.nanmin(channel), vmax=np.nanmax(channel))

    fig, (ax_path, ax_chan) = plt.subplots(1, 2, figsize=figsize)

    # --- left panel: path colored by channel ---
    segs = _segments(x, y)
    seg_vals = (channel[:-1] + channel[1:]) / 2.0  # value per segment
    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(seg_vals)
    lc.set_linewidth(2.5)
    ax_path.add_collection(lc)

    ax_path.scatter(x, y, c=channel, cmap=cmap, norm=norm, s=18,
                     edgecolors="none", zorder=3)
    ax_path.scatter(x[0], y[0], s=90, facecolors="none",
                     edgecolors="green", linewidths=2, zorder=4, label="start")
    ax_path.scatter(x[-1], y[-1], s=90, facecolors="none",
                     edgecolors="blue", linewidths=2, zorder=4, label="end")

    ax_path.set_xlim(x.min() - 0.02, x.max() + 0.02)
    ax_path.set_ylim(y.min() - 0.02, y.max() + 0.02)
    if invert_y:
        ax_path.invert_yaxis()
    ax_path.set_aspect("equal")
    ax_path.set_title("path (colored by {})".format(channel_label))
    # ax_path.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # --- right panel: channel vs index/time ---
    points = np.column_stack([xvals, channel]).reshape(-1, 1, 2)
    line_segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc2 = LineCollection(line_segs, cmap=cmap, norm=norm)
    lc2.set_array(seg_vals)
    lc2.set_linewidth(2.5)
    ax_chan.add_collection(lc2)
    ax_chan.scatter(xvals, channel, c=channel, cmap=cmap, norm=norm, s=18,
                     edgecolors="none", zorder=3)
    ax_chan.set_xlim(xvals.min(), xvals.max())
    pad = 0.05 * (np.nanmax(channel) - np.nanmin(channel) + 1e-9)
    ax_chan.set_ylim(np.nanmin(channel) - pad, np.nanmax(channel) + pad)
    ax_chan.set_xlabel("time (ms)" if t is not None else "sample index")
    ax_chan.set_ylabel(channel_label)
    ax_chan.set_title("{} over stroke".format(channel_label))

    if ref_line is not None:
        ax_chan.axhline(ref_line, color="gray", linestyle="--", linewidth=1)
        if ref_label:
            ax_chan.text(xvals.min(), ref_line, " " + ref_label,
                         va="bottom", ha="left", fontsize=8, color="gray")

    fig.colorbar(lc, ax=[ax_path, ax_chan], orientation="vertical",
                 fraction=0.03, pad=0.02, label=channel_label)

    if title:
        fig.suptitle(title)

    return fig, (ax_path, ax_chan)


def local_speed(x, y, t=None):
    """
    Per-sample speed proxy. If t is given, true speed (units/ms);
    otherwise point-to-point distance (valid as a speed proxy only when
    dt is roughly constant, as in your ~8-13ms capture).
    Returned array has same length as x/y (first sample repeats second).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.hypot(np.diff(x), np.diff(y))
    if t is not None:
        t = np.asarray(t, dtype=float)
        dt = np.diff(t)
        dt[dt == 0] = np.nan
        d = d / dt
    speed = np.empty_like(x)
    speed[1:] = d
    speed[0] = d[0] if len(d) else 0.0
    return speed


def local_curvature(x, y, signed=True):
    """
    Per-sample turning angle (radians) between the incoming and outgoing
    segment at each interior point: heading(P_i -> P_i+1) minus
    heading(P_i-1 -> P_i), wrapped to [-pi, pi].

    This is deliberately a *turning angle*, not a Menger/arc-length
    curvature: dividing by segment length blows up whenever two samples
    happen to be very close together (exactly what happens at a
    slow-moving hook apex), which makes that formulation numerically
    unstable right where you most want to look. Turning angle stays
    well-behaved regardless of local point spacing, and a hook/loop
    still shows up clearly as a large |angle| (approaching +-pi for a
    near-reversal, i.e. the tip of a hook or the self-crossing of a loop).

    signed=True keeps the turn direction (+ = CCW, - = CW, in standard
    math orientation on the x,y you pass in -- flip sign yourself if
    you've inverted y for display). signed=False returns |angle|, which
    is usually what you want for a "how sharp is this?" colormap.

    First and last samples are padded with 0 (no direction change is
    defined at the very ends of the stroke).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    heading = np.arctan2(np.diff(y), np.diff(x))  # one per segment, length n-1
    dtheta = np.diff(heading)  # one per interior point, length n-2
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    curvature = np.zeros(n)
    curvature[1:-1] = dtheta
    if not signed:
        curvature = np.abs(curvature)
    return curvature


if __name__ == "__main__":
    # 0: timestamp
    # 1: x
    # 2: y
    # 3: pressure
    # 4: pen
    # raw = np.load('./data/npy.5-raw/katakana_49/U+30A6.npy')
    # code_point = "U+30A6" # ウ
    # code_point = "U+30BB" # セ
    code_point = "U+7530" # 田
    # raw = np.load(f'./data/npy.5-raw/katakana_49/{code_point}.npy')
    raw = np.load(f'./data/npy.5-raw/kanken-10_80/{code_point}.npy')
    splits = np.where(raw[:, 4] == 0)[0] + 1
    strokes = np.split(raw[:,:-1], splits[:-1])
    # print(strokes)

    stroke_index = 2
    stroke = strokes[stroke_index]
    x = stroke[:, 1]
    y = stroke[:, 2]
    t = stroke[:, 0]
    pressure = stroke[:, 3]


    title = f"{code_point}, stroke: {stroke_index+1} (vertical-top)"

    floor = 0.08
    fig, _ = plot_stroke_diagnostic(
        x, y, pressure,
        t=t,
        channel_label="pressure (norm.)",
        ref_line=floor,
        ref_label="floor (1/3 raw)",
        title=title
    )
    # fig.tight_layout()
    fig.savefig("demo_pressure_colored_stroke.png", dpi=150)

    speed = local_speed(x, y, t)
    fig2, _ = plot_stroke_diagnostic(
        x, y, speed,
        t=t,
        channel_label="speed proxy",
        title=title,
    )
    # fig2.tight_layout()
    fig2.savefig("demo_speed_colored_stroke.png", dpi=150)

    curvature = local_curvature(x, y, signed=True)
    fig3, _ = plot_stroke_diagnostic(
        x, y, curvature,
        t=t,
        channel_label="turning angle (rad)",
        cmap="coolwarm",  # diverging, since curvature is signed
        ref_line=0.0, ref_label="straight",
        title=title,
    )
    # fig3.tight_layout()
    fig3.savefig("demo_curvature_colored_stroke.png", dpi=150)

    print("saved demo images")