#!/usr/bin/env python3
import sys
from signal import signal, SIGINT
from functools import partial

from kanji_nn.predef import compose, tap
from kanji_nn.etlcdb import glyph_iterator
import kanji_nn.etlcdb as etlcdb
import kanji_nn.bezier as bezier
from kanji_nn.bezier.spline_array import paths_to_array
from kanji_nn.bezier import render_fixed_ds_spline


def show(glyph):
    image = glyph["image:overlay"]
    image.show()
    print(glyph)
    input("...")


def save(glyph, key="image"):
    id = glyph["id"]
    literal = glyph["literal"]
    image = glyph[key]
    image.save(f"data/images/{id}.png")


def xxx(glyph):
    paths = glyph["kvg:paths"]
    spline = paths_to_array(paths)
    spline_image = render_fixed_ds_spline(spline, ds=1.0)
    print(spline_image)
    return glyph


pipeline = compose(
    tap(show),
    xxx,
    bezier.kvg_bbox,
    bezier.kvg_inject,
    etlcdb.overlay_skeleton,
    etlcdb.zhang_skeleton,
    # tap(partial(save, key="image:binary")),
    # tap(lambda x: print(x))
)

filters = compose(
    etlcdb.flag_border_touch,
    etlcdb.otsu,
)


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())


    query = """
        SELECT id, entry, literal, unicode, groups, data
        FROM   glyph
        WHERE  entry LIKE 'ETL1/%'
        AND    groups = 'KATAKANA'
        AND    literal = 'ア'
        ORDER  BY literal
    """

    query = """
        SELECT id, entry, literal, unicode, groups, data
        FROM   glyph
        WHERE  entry LIKE 'ETL1/%'
        AND    groups = 'KATAKANA'
        ORDER  BY literal
    """

    for glyph in glyph_iterator(query):
        glyph = filters(glyph)
        if glyph["skip"]:
            continue
        pipeline(glyph)
