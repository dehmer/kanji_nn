from PIL import Image


def splines_overlay(glyph):
    base = glyph["image:binary"].convert("RGB")
    mask = glyph["image:splines"].convert("L")
    green = Image.new("RGB", base.size, (0, 255, 0))
    return Image.composite(green, base, mask)
