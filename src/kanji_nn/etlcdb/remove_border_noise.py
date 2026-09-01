import numpy as np
from PIL import Image, ImageDraw


def split_features(feature_matrix: np.ndarray) -> list[np.ndarray]:
    """Split a feature matrix into a list of feature vectors."""
    return [row for row in feature_matrix]


def union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    min_x = min(a[0], b[0])
    min_y = min(a[1], b[1])
    max_x = max(a[2], b[2])
    max_y = max(a[3], b[3])

    return np.array([
        min_x, min_y, max_x, max_y,
        (max_x - min_x) * (max_y - min_y),
        a[5] + b[5]
    ])


def intersects(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    padding: int = 0,
) -> bool:
    a_min_x, a_min_y, a_max_x, a_max_y = feature_a[:4]
    b_min_x, b_min_y, b_max_x, b_max_y = feature_b[:4]

    return (
        a_min_x - padding <= b_max_x + padding
        and a_max_x + padding >= b_min_x - padding
        and a_min_y - padding <= b_max_y + padding
        and a_max_y + padding >= b_min_y - padding
    )


def draw_feature_bboxes(features, size) -> Image.Image:
    image = Image.new("RGB", size, "black")

    draw = ImageDraw.Draw(image)
    for feature in features:
        min_x, min_y, max_x, max_y = feature[:4]
        points = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ]
        draw.line(points, fill="white", width=1)

    return image


def fold_connected(
    features: list[np.ndarray],
    padding: int = 0,
) -> list[np.ndarray]:
    features = features.copy()

    changed = True
    while changed:
        changed = False

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if intersects(features[i], features[j], padding):
                    merged = union(features[i], features[j])

                    # Remove both and insert their union.
                    features.pop(j)
                    features.pop(i)
                    features.append(merged)

                    changed = True
                    break

            if changed:
                break

    return features


def is_near_border(size, feature, margin):
    x_min, y_min, x_max, y_max = feature[:4]
    return (
        x_min < margin
        or y_min < margin
        or x_max > size[0] - margin
        or y_max > size[1] - margin
    )


def remove_border_noise(glyph, margin=2):
    labels, features = glyph["labels"], glyph["features"]
    size = glyph["size"]

    data = np.array(glyph["image:binary"])

    features = split_features(features)
    features = fold_connected(features, padding=0)
    features_image = draw_feature_bboxes(features, glyph["size"])

    for feature in features:
        x_min, y_min, x_max, y_max, area, num_pixels = feature
        density = num_pixels / area
        # density threshold is empirical:
        if density >= 0.4 and is_near_border(size, feature, margin):
            data[y_min : y_max, x_min : x_max] = False

    binary_image = Image.fromarray(data)

    return glyph | {"image:features": features_image, "image:binary": binary_image}
