import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_stroke_geometry(geometries,
                         figsize=(14, 3.5),
                         cmap="turbo",
                         curvature_percentile=95):
    """
    Plot one or more strokes together with their geometric properties.

    Parameters
    ----------
    geometries : list of dict
        Result of stroke_geometry().
    figsize : tuple
        Figure size per row.
    cmap : str
        Matplotlib colormap.
    curvature_percentile : float
        Percentile used to normalize curvature colors.
        Prevents a single spike from dominating the color scale.
    """

    n = len(geometries)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(figsize[0], figsize[1] * n),
        constrained_layout=True,
        squeeze=False,
    )

    for i, g in enumerate(geometries):
        ax0 = axes[i, 0]
        ax1 = axes[i, 1]

        xy = g["xy"]
        s = g["norm(s)"]
        curvature = g["gauss:K"]

        # ------------------------------------------------------
        # Left plot: stroke colored by curvature
        # ------------------------------------------------------

        points = xy.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Use curvature at the first point of each segment
        c = curvature[:-1]

        vmax = np.percentile(curvature, curvature_percentile)
        vmax = max(vmax, 1e-6)

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=Normalize(vmin=0.0, vmax=vmax),
        )

        lc.set_array(c)
        lc.set_linewidth(2.5)

        ax0.add_collection(lc)

        # ax0.scatter(*xy[0], c="limegreen", s=60, zorder=3, label="start")
        # ax0.scatter(*xy[-1], c="red", s=60, zorder=3, label="end")

        ax0.autoscale()
        ax0.set_aspect("equal")
        ax0.invert_yaxis()
        ax0.set_title(f"Stroke {i}")
        # ax0.legend(loc="best")

        cb = fig.colorbar(lc, ax=ax0, fraction=0.046, pad=0.04)
        cb.set_label("Curvature")

        # ------------------------------------------------------
        # Right plot: geometry
        # ------------------------------------------------------

        ax1.plot(s, g["gauss:θ"], lw=2, label="gauss:θ")
        ax1.plot(s, curvature, lw=2, label="curvature")

        if "gauss:dθ/ds" in g:
            ax1.plot(s, g["gauss:dθ/ds"], "--", lw=1.5, label="gauss:dθ/ds")

        ax1.set_xlim(0.0, 1.0)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("Normalized arc length")
        ax1.set_title("Geometry")
        ax1.legend(loc="best")

    plt.show()