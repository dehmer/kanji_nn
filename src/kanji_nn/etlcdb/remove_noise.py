import numpy as np
from PIL import Image


def remove_noise(glyph, min_size=5):
    """
    Finds and deletes white pixel clusters smaller than min_size.
    """
    labels, features = glyph["labels"], glyph["features"]
    data = labels > 0
    num_pixels = features[:, -1]

    indices = np.where(num_pixels < min_size)[0]
    indices = indices + 1 # compensate for background (label: 0)
    noise_mask = np.isin(labels, indices)
    data[noise_mask] = 0
    image = Image.fromarray((data * 255).astype(np.uint8)).convert("1")

    return glyph | {"image:binary": image}
