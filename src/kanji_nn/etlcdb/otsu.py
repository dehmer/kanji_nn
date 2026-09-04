from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu


def otsu(glyph, noise_threshold=0.5):
    image = glyph["image"]

    if image.mode == "1":
        binary = image
    else:
        data = np.array(image)
        threshold = threshold_otsu(data)
        binary = image.point(lambda p: 255 if p > threshold else 0)

    histogram = binary.histogram()
    noise_ratio = histogram[255] / histogram[0]
    if noise_ratio > noise_threshold:
        return glyph | {"skip": True, "reason": f"[otsu] glyph too noisy; ratio={noise_ratio}"}

    return glyph | {"image:binary": binary}
