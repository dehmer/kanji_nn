import numpy as np
from skan.csr import Skeleton
from .connected_features import fold


def skeleton_graph(glyph):
    image = np.asarray(glyph["image:skeleton"]) > 0

    # Find and merge 2x2 pixel clusters:
    ambiguities = fold([
        [c, r, c + 2, r + 2]
        for r, c in zip(*np.where(image[:-1, :-1]))
        if image[r:r+2, c:c+2].all()
    ])

    if len(ambiguities):
        return glyph | {"skip": True, "reason": "ambiguous skeletonization (2x2 clusters)"}

    skeleton = Skeleton(image)
    return glyph | {"skeleton": skeleton}
