

def flag_label_mismatch(glyph):
    """
    Flag glyphs where literal does not match the actual image.
    The corresponsing ids are collected during manual inspection/eyeballing.
    """
    ids = [
        "a98b65e7-f2ce-4c2c-b354-6d93658993f6"
    ]

    if glyph["id"] in ids:
        return glyph | {"skip": True, "reason": "label mismatch"}
    else:
        return glyph
