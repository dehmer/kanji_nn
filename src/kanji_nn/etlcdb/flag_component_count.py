from .extract_features import extract_features


def flag_component_count(glyph):
    num_strokes = glyph["num_strokes"]
    _, features = extract_features(glyph["image:binary"])

    # Feature count may be less, but never more than stroke count:
    if (len(features) > num_strokes):
        return glyph | {"skip": True, "reason": "too many feature detected"}

    return glyph
