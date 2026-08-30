from PIL import Image


def overlay_splines(glyph):
    binary_image = glyph["image:binary"].convert("RGB")
    splines_image = glyph["image:splines"]
    mask = splines_image.convert("L")  # black bg -> 0, green strokes -> nonzero
    overlay = Image.composite(splines_image, binary_image, mask)
    return glyph | {"overlay:splines": overlay}
