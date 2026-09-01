import numpy as np
from PIL import Image
import scipy.ndimage as ndimage


def extract_features(glyph):
    """
    Extract labels and features from binary image.
    """
    binary_image = glyph["image:binary"]
    data = np.array(binary_image) > 0
    labels, _ = ndimage.label(data)

    num_pixels = np.bincount(labels.ravel()) # 0: background pixels
    slices = ndimage.find_objects(labels)

    # feature :: [x_min, y_min, x_max, y_max, bbox_area, num_pixels]
    # features :: [feature]
    features = []
    for i, slice_ in enumerate(slices):
        if slice_ is None:
            continue

        row, col = slice_
        y_min, y_max = row.start, row.stop
        x_min, x_max = col.start, col.stop
        width, height = x_max - x_min, y_max - y_min
        area = width * height
        feature = np.asarray([x_min, y_min, x_max, y_max, area, num_pixels[i + 1]])
        features.append(feature)

    features = np.vstack(features)
    return glyph | {"labels": labels, "features": features}
