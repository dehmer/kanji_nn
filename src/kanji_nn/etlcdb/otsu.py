from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi


def otsu(glyph, fill_threshold=0.5):
    image = glyph["image"]

    if image.mode == "1":
        binary = image
    else:
        data = np.array(image)
        threshold = threshold_otsu(data)
        binary = image.point(lambda p: 255 if p > threshold else 0)

    binary_mask = np.asarray(binary) > 0
    foreground = binary_mask.sum()
    background = binary_mask.size - foreground
    fill_ratio = foreground / background if background else np.inf

    if fill_ratio > fill_threshold:
        return glyph | {"skip": True, "reason": f"[otsu] glyph too noisy; ratio={fill_ratio:.3f}"}

    # Calculate euclidean distance transform (EDT):
    # Straight-line distance from every foreground pixel to
    # the nearest background pixel.
    #
    edt = ndi.distance_transform_edt(binary_mask)

    return glyph | {"image:binary": binary, "edt": edt}
