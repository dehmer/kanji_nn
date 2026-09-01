from PIL import Image


def skeleton_overlay(glyph):
    base = glyph["image:binary"].convert("RGB")
    mask = glyph["image:skeleton"] # mode "L", 0/255
    red = Image.new("RGB", base.size, (255, 0, 0))
    return Image.composite(red, base, mask)
