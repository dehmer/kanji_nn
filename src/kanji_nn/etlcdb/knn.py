import sys
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


def plot_stroke_assignment(glyph, ambiguous_margin=0.2):
    xysp = glyph["splines:xysp"]
    coordinates = glyph["skeleton"].coordinates
    stroke = glyph["knn:stroke"]
    margin = glyph["knn:margin"]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Reference strokes, faint, for context.
    sp = xysp[:, 2:]
    stroke_ids = np.cumsum(sp[:, 1] == 0) - (sp[:, 1] == 0)
    for sid in np.unique(stroke_ids):
        pts = xysp[stroke_ids == sid, :2]
        ax.plot(pts[:, 0], pts[:, 1], "-", color="lightgray", linewidth=1, zorder=1)

    # Skeleton points, colored by winning stroke assignment.
    winning_stroke = stroke[:, 2]
    ax.scatter(coordinates[:, 1], coordinates[:, 0], c=winning_stroke,
               cmap="tab10", s=15, zorder=2)

    # Low-margin (ambiguous) points, ringed.
    ambiguous = margin[:, 2] < ambiguous_margin
    ax.scatter(coordinates[ambiguous, 1], coordinates[ambiguous, 0],
               facecolors="none", edgecolors="black", s=60, zorder=3)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"{glyph.get('literal', '')} — stroke assignment")

    plt.show()
    return glyph


def plot_margin_histogram(glyph, bins=50):
    import matplotlib.pyplot as plt

    margin = glyph["knn:margin"][:, 2]
    stroke = glyph["knn:stroke"]
    agreement = stroke[:, 4].astype(bool)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(margin[agreement], bins=bins, alpha=0.6, label="agree", color="tab:blue")
    ax.hist(margin[~agreement], bins=bins, alpha=0.6, label="disagree", color="tab:red")
    ax.set_xlabel("margin (dist[1] - dist[0])")
    ax.set_ylabel("count")
    ax.set_title(f"{glyph.get('literal', '')} — margin distribution")
    ax.legend()
    plt.show()
    return glyph


def knn(glyph):
    xysp = glyph["splines:xysp"]
    skeleton = glyph["skeleton"]
    coordinates = skeleton.coordinates

    # Keep stroke/segment indices separate:
    sp = xysp[:, 2:]
    ssp = np.column_stack((np.cumsum(sp[:, 1] == 0) - (sp[:, 1] == 0), sp))

    kvg_tree = KDTree(xysp[:, :-2])
    distance, neighbor = kvg_tree.query(coordinates, k=2)
    stroke = np.column_stack((neighbor, ssp[neighbor, 0]))

    # 1: same stroke, 0: different stroke
    agreement = stroke[:, 2] == stroke[:, 3]
    stroke = np.column_stack((stroke, agreement))
    disagreement = np.where(stroke[:, 4] == 0)[0]
    margin = np.column_stack((distance, distance[:, 1] - distance[:, 0]))

    return glyph | {"knn:stroke": stroke, "knn:margin": margin}