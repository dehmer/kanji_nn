from PIL import Image
import numpy as np
from skimage.morphology import skeletonize


def zhang_skeleton(glyph):
    image = glyph["image:binary"]
    stroke_mask = np.array(image.convert("L")) > 0
    skeleton_mask = skeletonize(stroke_mask, method="zhang")
    skeleton_uint8 = np.where(skeleton_mask, 255, 0).astype(np.uint8)
    skeleton_image = Image.fromarray(skeleton_uint8)

    return glyph | {"image:skeleton": skeleton_image}
