import numpy as np
from skan.csr import Skeleton
from scipy import ndimage as ndi
import kanji_nn.etlcdb.skeleton_ops as sops
from .connected_features import fold


def skeleton_graph(glyph):
    skeleton_image = np.asarray(glyph["image:skeleton"])
    skeleton_mask = skeleton_image > 0

    # Find and merge 2x2 pixel clusters:
    ambiguities = fold([
        [c, r, c + 2, r + 2]
        for r, c in zip(*np.where(skeleton_mask[:-1, :-1]))
        if skeleton_mask[r:r+2, c:c+2].all()
    ])

    if len(ambiguities):
        return glyph | {"skip": True, "reason": "ambiguous skeletonization (2x2 clusters)"}

    skeleton = Skeleton(skeleton_mask)
    paths = [skeleton.path(i) for i in range(0, skeleton.n_paths)]

    return glyph | {
        "skeleton:paths": paths,

        # flip row/column layout to x/y layout:
        "skeleton:xy": skeleton.coordinates[:, ::-1],
        "skeleton:degrees": skeleton.degrees,
        "skeleton:lengths": skeleton.path_lengths()
    }
