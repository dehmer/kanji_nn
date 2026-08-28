from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu


def otsu(glyph):
    image = glyph["image"]

    if image.mode == "1":
        return glyph | {"image:binary": image}

    data = np.array(image)
    threshold = threshold_otsu(data)
    binary = image.point(lambda p: 255 if p > threshold else 0)

    return glyph | {"image:binary": binary}
