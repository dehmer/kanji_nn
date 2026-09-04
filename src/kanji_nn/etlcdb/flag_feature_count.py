import numpy as np
from .extract_features import extract_features
from .connected_features import connected_features


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
