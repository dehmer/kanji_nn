from PIL import Image
import numpy as np


def overlay_skeleton(glyph):
    base = glyph["image:binary"].convert("RGB")
    skeleton = glyph["image:skeleton"]  # mode "L", 0/255

    red = Image.new("RGB", base.size, (255, 0, 0))
    overlay = Image.composite(red, base, skeleton)

    return glyph | {"overlay:skeleton": overlay}
