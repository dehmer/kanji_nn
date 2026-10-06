import numpy as np
from skan.csr import Skeleton
from scipy import ndimage as ndi
import kanji_nn.etlcdb.skeleton_ops as sops
from .connected_features import fold


def skeleton_graph(glyph):
    mask = glyph["skeleton:mask"]

    # Find and merge 2x2 pixel clusters:
    ambiguities = fold([
        [c, r, c + 2, r + 2]
        for r, c in zip(*np.where(mask[:-1, :-1]))
        if mask[r:r+2, c:c+2].all()
    ])

    if len(ambiguities):
        return glyph | {"skip": True, "reason": "ambiguous skeletonization (2x2 clusters)"}

    skeleton = Skeleton(mask)
    paths = [skeleton.path(i) for i in range(0, skeleton.n_paths)]

    edt = glyph["edt"] # complete EDT field
    degrees = skeleton.degrees
    xy = skeleton.coordinates[:, ::-1] # flip row/column layout to x/y layout
    edt = edt[xy[:, 0], xy[:, 1]] # per-pixel EDT

    # Highly unlikely, but check anyway if skeleton has no inner pixels:
    if len(edt[degrees == 2]) == 0:
        return glyph | {"skip": True, "reason": "degenerated glyph; no inner skeleton pixels found"}

    return glyph | {
        "skeleton:paths": paths,
        "skeleton:xy": xy,
        "skeleton:degrees": degrees,
        "skeleton:lengths": skeleton.path_lengths(),
        "skeleton:edt:inner": np.median(edt[degrees == 2])
    }
