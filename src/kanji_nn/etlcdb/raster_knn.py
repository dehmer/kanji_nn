import sys
import numpy as np
from scipy.spatial import KDTree
import kanji_nn.bezier as bezier


def stroke_slices(splines):
    """Row index ranges [start, stop) for each stroke, from pen==0 markers."""
    n = splines.shape[0]
    ends = np.where(splines[:, 8] == 0)[0] + 1
    starts = np.concatenate(([0], ends[:-1]))
    return list(zip(starts, ends))


def relocate_handles(stroke, new_anchors):
    """
    Relocate inner Bézier handles (p1, p2) for a single stroke's segments,
    given the stroke's original segments and its n+1 new anchor points.
    """
    eps = 1e-9

    def normalize(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.divide(v, n, out=np.zeros_like(v), where=n > eps)

    p0, p1, p2, p3 = stroke[:, 0:2], stroke[:, 2:4], stroke[:, 4:6], stroke[:, 6:8]
    n = stroke.shape[0]

    h_out, h_in = p1 - p0, p3 - p2
    mag_out, mag_in = np.linalg.norm(h_out, axis=1), np.linalg.norm(h_in, axis=1)
    dir_out, dir_in = normalize(h_out), normalize(h_in)

    old_anchors = np.vstack([p0, p3[-1]])
    new_anchors = np.asarray(new_anchors, dtype=float)

    L_old = np.linalg.norm(np.diff(old_anchors, axis=0), axis=1)
    L_new = np.linalg.norm(np.diff(new_anchors, axis=0), axis=1)
    ratio = np.divide(L_new, L_old, out=np.ones_like(L_old), where=L_old > eps)

    dir_start, dir_end = dir_out.copy(), dir_in.copy()
    for k in range(1, n):  # interior anchors only; whole stroke is one smooth chain
        bisector = normalize(dir_in[k - 1] + dir_out[k])
        dir_end[k - 1] = bisector
        dir_start[k] = bisector

    new_stroke = stroke.copy()
    new_stroke[:, 0:2] = new_anchors[:-1]
    new_stroke[:, 2:4] = new_anchors[:-1] + (ratio * mag_out)[:, None] * dir_start
    new_stroke[:, 4:6] = new_anchors[1:] - (ratio * mag_in)[:, None] * dir_end
    new_stroke[:, 6:8] = new_anchors[1:]
    return new_stroke


def raster_knn(glyph):
    """
    Naive nearest-neighbor matching: snap each spline anchor to its
    nearest pixel in the binary image.

    Two failure modes follow directly from the "nearest" in nearest-
    neighbor. First, correspondence is per-anchor and unaware of stroke
    identity, so anchors of a short or misplaced spline may snap to a
    pixel stroke other than their intended one. Second, since anchors
    only ever move to an existing pixel, a spline can't stretch beyond
    its original length — an already well-aligned but too-short spline
    stays short, with its outer anchors never reaching the stroke's
    true extent.

    Despite this, results are surprisingly good whenever bounding box,
    orientation, and per-stroke spatial distribution and length are
    already close to the target.
    """

    splines = glyph["splines"]
    image = glyph["image:binary"]
    mask = np.array(image) > 0
    image_pts = np.argwhere(mask)[:, ::-1]
    pixel_tree = KDTree(image_pts)

    stick_man, distances_all, neighbors_all, new_strokes = [], [], [], []

    for start, stop in stroke_slices(splines):
        stroke = splines[start:stop]
        anchors = np.vstack([stroke[:, :2], stroke[-1, 6:8]])
        distances, neighbors = pixel_tree.query(anchors)
        new_anchors = image_pts[neighbors]

        new_strokes.append(relocate_handles(stroke, new_anchors))
        stick_man.append(new_anchors)
        distances_all.append(distances)
        neighbors_all.append(neighbors)

    new_splines = np.vstack(new_strokes)
    xysp = bezier.resample_splines(new_splines, ds=0.5)
    split_indices = np.where(xysp[:, 3] == 0)[0] + 1
    xysp = np.split(xysp[:, :2], split_indices[:-1])

    return glyph | {
        "splines": np.vstack(new_strokes),
        # "stick_man": stick_man,
        "stick_man": xysp,
        "stick_man:distances": np.concatenate(distances_all),
        "stick_man:neighbors": np.concatenate(neighbors_all),
    }