import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from .connected_features import connected_features


def features_from_slices(slices):
    features = []
    for i, slice_ in enumerate(slices):
        if slice_ is None:
            continue

        row, col = slice_
        y_min, y_max = row.start, row.stop
        x_min, x_max = col.start, col.stop
        feature = np.asarray([x_min, y_min, x_max, y_max])
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


def detect_noise(glyph, min_size=5, margin=2, padding=2):
    binary_image = glyph["image:binary"]

    # boolean mask: height rows x width columns
    # True: pixel, False: background
    size = glyph["size"]
    mask = np.array(binary_image) > 0

    labels, num_features = ndimage.label(mask)

    # pixel count per label, incl. background (label 0):
    num_pixels = np.bincount(labels.ravel()) # 0: background pixels
    indices = np.where(num_pixels < min_size)[0]
    noise_field = np.isin(labels, indices)

    # x/y slice per feature, excl. background:
    slices = ndimage.find_objects(labels)
    features = features_from_slices(slices)
    features = np.delete(features, indices - 1, axis=0)
    connected = np.vstack(connected_features(features, padding=padding))

    noise_boxes = []
    for feature in connected:
        x_min, y_min, x_max, y_max = feature
        touches = border_touches(size, feature, margin)
        ratios = [touch / size[i % 2] for i, touch in enumerate(touches)]
        ratio = max(ratios)

        if ratio > 0.5: continue
        elif ratio == 0.0: continue
        else: noise_boxes.append([y_min, y_max, x_min, x_max])

    return glyph | {
        "noise:field": noise_field,
        "noise:boxes": noise_boxes
    }


def remove_noise(glyph):
    binary_image = glyph["image:binary"]
    image = glyph["image"] # grayscale image
    noise_field = glyph["noise:field"]
    noise_boxes = glyph["noise:boxes"]

    # boolean mask: height rows x width columns
    # True: pixel, False: background
    binary_mask = np.array(binary_image) > 0
    image_data = np.array(image)

    binary_mask[noise_field] = False
    image_data[noise_field] = 0 # background = 0

    for box in noise_boxes:
        y_min, y_max, x_min, x_max = box
        binary_mask[y_min : y_max, x_min : x_max] = False
        image_data[y_min : y_max, x_min : x_max] = 0

    binary_image = Image.fromarray(binary_mask)
    image = Image.fromarray(image_data)

    return glyph | {
        "image": image,
        "image:binary": binary_image
    }
