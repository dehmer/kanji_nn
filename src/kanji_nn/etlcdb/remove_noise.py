import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from .connected_features import connected_features


def features_from_slices(slices, num_pixels):
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

    return np.vstack(features)


def border_touches(size, feature, margin=0):
    """
    Check if bounding box touches with either border.
    Return array for left, top, right bottom border with:
    - 0: Does not touch.
    - else: Width/height in pixels in direction opposite to border.
    """
    # x_min, y_min: inclusive
    # x_max, y_max: exclusive
    x_min, y_min, x_max, y_max = feature[:4]
    return [
        int(x_max) if x_min <= margin else 0,
        int(y_max) if y_min <= margin else 0,
        int(size[0] - x_min) if x_max >= size[0] - margin else 0,
        int(size[1] - y_min) if y_max >= size[1] - margin else 0
    ]


def remove_noise(glyph, min_size=5, margin=2, padding=2):
    image = glyph["image:binary"]

    # boolean mask: height rows x width columns
    # True: pixel, False: background
    size = glyph["size"]
    data = np.array(image) > 0

    labels, num_features = ndimage.label(data)

    # pixel count per label, incl. background (label 0):
    num_pixels = np.bincount(labels.ravel()) # 0: background pixels
    indices = np.where(num_pixels < min_size)[0]
    noise = np.isin(labels, indices)
    data[noise] = False

    # x/y slice per feature, excl. background:
    slices = ndimage.find_objects(labels)
    features = features_from_slices(slices, num_pixels)
    features = np.delete(features, indices - 1, axis=0)
    connected = np.vstack(connected_features(features, padding=padding))

    for feature in connected:
        x_min, y_min, x_max, y_max, _, _ = feature
        touches = border_touches(size, feature, margin)
        ratios = [touch / size[i % 2] for i, touch in enumerate(touches)]
        ratio = max(ratios)

        if ratio > 0.5: continue
        elif ratio == 0.0: continue
        else: data[y_min : y_max, x_min : x_max] = False


    image = Image.fromarray((data * 255).astype(np.uint8)).convert("1")
    return glyph | {"image:binary": image}
