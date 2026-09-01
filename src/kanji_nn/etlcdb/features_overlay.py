from PIL import Image


def features_overlay(glyph):
    base = glyph["image:binary"].convert("RGB")
    mask = glyph["image:features"].convert("L")
    blue = Image.new("RGB", base.size, (0, 255, 0))
    return Image.composite(blue, base, mask)
