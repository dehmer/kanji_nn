import numpy as np
from PIL import Image, ImageDraw


def draw_splines(xyp: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """
    Draw resampled points (M,3): x, y, pen onto a new PIL image.
    Background black, strokes drawn in `color`. pen=0 ends a spline.
    """
    image = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(image)

    split_indices = np.where(xyp[:, -1] == 0)[0] + 1
    spline_list = np.split(xyp[:, :2], split_indices[:-1])

    for spline in spline_list:
        points = [tuple(p) for p in spline]
        draw.line(points, fill="white", width=1)

    return image


def splines_overlay(glyph):
    base = glyph["image:binary"].convert("RGB")

    xysp = glyph["splines:xysp"]
    xyp = xysp[:, [0,1,3]]
    splines_image = draw_splines(xyp, glyph["size"])
    mask = splines_image.convert("L")
    green = Image.new("RGB", base.size, (0, 255, 0))

    return Image.composite(green, base, mask)
