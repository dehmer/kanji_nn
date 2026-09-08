import numpy as np
import kanji_nn.bezier as bezier


def resample_splines(glyph, ds=0.5):
    splines = glyph["splines"]

    # [x, y, segment index, pen-down/-up]
    xysp = bezier.resample_splines(splines, ds=ds)
    return glyph | {"splines:xysp": xysp}
