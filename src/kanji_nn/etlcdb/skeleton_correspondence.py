import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize_scalar
import sys
import matplotlib.pyplot as plt


def plot_stroke_assignments(glyph, assignments, num_strokes):
    skeleton_image = np.asarray(glyph["image:skeleton"])
    target_xy = glyph["skeleton"].coordinates[:, ::-1]  # x/y, matches assignment target index

    assignments = np.array(assignments, dtype=object)
    stroke_of = assignments[:, 1].astype(int)
    distance_of = assignments[:, 4].astype(float)

    fig, axes = plt.subplots(2, (num_strokes + 1) // 2, figsize=(4 * ((num_strokes + 1) // 2), 8))
    axes = axes.ravel()

    for stroke_id in range(num_strokes):
        ax = axes[stroke_id]
        ax.imshow(skeleton_image, cmap="gray")

        mask = stroke_of == stroke_id
        pts = target_xy[mask]
        dist = distance_of[mask]

        sc = ax.scatter(pts[:, 0], pts[:, 1], c=dist, cmap="viridis", s=8)
        ax.set_title(f"stroke {stroke_id} (n={mask.sum()})")
        ax.axis("off")

    for ax in axes[num_strokes:]:
        ax.axis("off")

    fig.colorbar(sc, ax=axes.tolist(), label="fit distance", shrink=0.6)
    plt.show()
    return fig


def bezier_point(ctrl, t):
    p0, p1, p2, p3 = ctrl
    mt = 1.0 - t
    return (mt**3) * p0 + 3 * (mt**2) * t * p1 + 3 * mt * (t**2) * p2 + (t**3) * p3


def closest_point_on_segment(point, ctrl, n_samples=20):
    """
    Coarse sample + local refine — a cheap stand-in for Schneider's
    subdivide/bisect scheme, adequate given the fit step doesn't need t
    to be exact, just close enough to seed the optimizer.

    Returns
    -------
    [t, distance]
    """
    ts = np.linspace(0.0, 1.0, n_samples)
    pts = np.array([bezier_point(ctrl, t) for t in ts])
    i0 = np.argmin(np.linalg.norm(pts - point, axis=1))
    lo, hi = ts[max(i0 - 1, 0)], ts[min(i0 + 1, n_samples - 1)]

    def dist_sq(t):
        return np.sum((bezier_point(ctrl, np.clip(t, 0.0, 1.0)) - point) ** 2)

    res = minimize_scalar(dist_sq, bounds=(lo, hi), method="bounded")
    return float(res.x), float(np.sqrt(res.fun))


def skeleton_correspondence(glyph):
    splines = glyph["splines"]
    xysp = glyph["splines:xysp"] # resampled (ds=0.5) with segment index and pen-down/-up
    skeleton = glyph["skeleton"]
    skeleton_image = np.asarray(glyph["image:skeleton"])
    paths = glyph["kvg:paths"]
    num_strokes = glyph["num_strokes"]
    pixel_graph = glyph["pixel_graph"]

    """
    Sample under observation: 例 - 2e767b06-9809-45e7-85c3-9a3a4495257d

    - num_strokes/paths (reference): 8 <- num_strokes, len(paths)
    - total segment count: 21 <- len(splines)
    - distinct skeleton branches: 13 <- skeleton.n_paths
    - distinct junctions: 5 <- len(np.where(skeleton.degrees > 2)[0])
    - skeleton pixels (target points): 235 <- len(skeleton.coordinates)
    """

    # Insert stroke and running segment index.
    # Also mitigate segment index/count mismatch:
    # Relabel SVG 'Move' segment from 0 to 1 and normalize back to 0.
    #
    # sp :: [segment-per-stroke, pen]
    # ssp :: [stroke, segment-per-stroke, pen]
    # ssi :: [stroke, segment-per-stroke, running-segment]
    sp = xysp[:, 2:]
    stroke = np.cumsum(sp[:, 1] == 0) - (sp[:, 1] == 0)
    ssp = np.column_stack([stroke, sp])
    ssp[np.where(ssp[:, 1] == 0)[0], 1] = 1
    ssp[:, 1] -= 1

    _, segment_idx = np.unique(ssp[:, 0:2], axis=0, return_inverse=True)
    ssi = np.column_stack((ssp[:, :-1], segment_idx))


    # TODO: Step 1 — closest point on a curve, for one target point.

    reference_xy = xysp[:, :-2]
    target_xy = skeleton.coordinates[:, ::-1] # flip row/column -> x/y
    kvg_tree = KDTree(reference_xy)
    distance, neighbor = kvg_tree.query(target_xy)
    nearest_segment = ssi[neighbor]

    # assignment :: (target, stroke, segment, t, distance, branch | np.nan)
    # assignments :: [assignment]
    #
    assignments = []
    for i in range(0, len(target_xy)):
        point = target_xy[i]
        stroke, segment, key = nearest_segment[i]
        ctrl = splines[int(key), :8].reshape(-1, 2)
        t, distance = closest_point_on_segment(point, ctrl)
        branches = pixel_graph.pixel_branches[i]
        branch_id = branches[0] if len(branches) == 1 else np.nan
        assignment = (i, int(stroke), int(segment), t, distance, branch_id)
        assignments.append(assignment)

    # assignments = np.array(assignments)
    # print(assignments)
    plot_stroke_assignments(glyph, assignments, num_strokes)

    # TODO: It would probably be interesting to compare KNN distance with assignment distance

    # TODO: Step 2 — aggregate correspondences per segment.
    # TODO: Step 3 — the redistribution fix (Figure 4).
    # TODO: Step 4 — the actual fit, per segment, via optimization.

    return glyph