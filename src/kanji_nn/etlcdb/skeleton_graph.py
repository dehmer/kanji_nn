import numpy as np
from skan.csr import Skeleton
from scipy import ndimage as ndi
from .connected_features import fold
from .pixel_graph import PixelGraph


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
    pixel_graph = PixelGraph(skeleton, glyph["edt"])

    return glyph | {"pixel_graph": pixel_graph}
