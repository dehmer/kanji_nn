

def flag_component_count(glyph):
    features = glyph["features"]
    num_strokes = glyph["num_strokes"]

    # Feature count may be less, but never more than stroke count:
    if (len(features) > num_strokes):
        return glyph | {"skip": True, "reason": "too many feature detected"}

    return glyph
