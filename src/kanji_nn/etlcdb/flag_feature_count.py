import numpy as np
import numpy as np
import scipy.ndimage as ndimage
from .connected_features import connected_features


def extract_features(binary_image):
    """
    Extract labels and features from binary image.
    """
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


    features = np.vstack(features) if len(features) else None
    return labels, features


def flag_feature_count(glyph, padding=0):
    num_strokes = glyph["num_strokes"]
    _, features = extract_features(glyph["image:binary"])

    if features is None:
        return glyph | {"skip": True, "reason": "no features detected"}

    features = connected_features(features, padding=padding)

    # Feature count may be less, but never more than stroke count:
    if len(features) > num_strokes:
        return glyph | {"skip": True, "reason": f"too many feature detected; {len(features)}/{num_strokes}"}

    else:
        return glyph
