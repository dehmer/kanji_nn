import sys
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from skan.csr import Skeleton


def plot_stroke_assignments(glyph):
    assignments = glyph["assignments"]
    num_strokes = glyph["num_strokes"]
    skeleton_image = np.asarray(glyph["image:skeleton"])
    mask = glyph["skeleton:mask"]
    skeleton = Skeleton(mask)

    target_xy = skeleton.coordinates[:, ::-1]
    stroke_of = assignments[:, 2].astype(int)
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
    plt.close(fig)
    return glyph


def skeleton_assignment(glyph):
    # resampled (ds=0.5) with segment index and pen-down/-up
    xysp = glyph["splines:xysp"]
    splines = glyph["splines"]
    paths = glyph["kvg:paths"]
    mask = glyph["skeleton:mask"]
    skeleton = Skeleton(mask)

    degrees = skeleton.degrees
    target_xy = skeleton.coordinates[:, ::-1]

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

    reference_xy = xysp[:, :-2]
    kvg_tree = KDTree(reference_xy)
    knn_distance, neighbor = kvg_tree.query(target_xy)
    nearest_segment = ssi[neighbor]

    # Reverse lookup: {pixel index: [path index]}.
    pixel_paths = defaultdict(list)
    for path_idx in range(0, skeleton.n_paths):
        pixel_indices = skeleton.path(path_idx)
        for pixel_idx in pixel_indices:
            pixel_paths[int(pixel_idx)].append(path_idx)

    # assignment :: [target, branch, stroke, segment, knn-distance]
    # assignments :: [assignment]
    #
    assignments = []
    for target_index in range(0, len(target_xy)):
        point = target_xy[target_index]
        stroke, segment, key = nearest_segment[target_index]
        ctrl = splines[int(key), :8].reshape(-1, 2)
        paths = pixel_paths[target_index]

        # Ignore target junction pixels (degree > 2):
        path = pixel_paths[target_index][0] if degrees[target_index] <= 2 else np.nan

        assignments.append([
            target_index,
            path,
            stroke,
            segment,
            knn_distance[target_index]
        ])


    assignments = np.array(assignments)
    return glyph | {"assignments": assignments}
