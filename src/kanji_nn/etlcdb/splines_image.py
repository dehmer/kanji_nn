import numpy as np
from PIL import Image, ImageDraw
import kanji_nn.bezier as bezier


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


def splines_image(glyph):
    splines = glyph["splines"]
    xysp = bezier.resample_splines(splines)
    xyp = xysp[:, [0,1,3]]
    splines_image = draw_splines(xyp, glyph["size"])
    return glyph | {"image:splines": splines_image}
