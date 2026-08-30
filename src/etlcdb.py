#!/usr/bin/env python3
import sys
from signal import signal, SIGINT
from functools import partial
import numpy as np
from PIL import Image

from kanji_nn.predef import compose, tap
from kanji_nn.etlcdb import glyph_iterator
import kanji_nn.etlcdb as etlcdb
import kanji_nn.bezier as bezier


def await_input(glyph):
    input("...")
    return glyph


def show(glyph, key="image"):
    image = glyph[key]
    image.show()


def save(glyph, key="image"):
    id = glyph["id"]
    literal = glyph["literal"]
    image = glyph[key]
    image.save(f"data/images/{id}.png")


pipeline = compose(
    # await_input,
    tap(partial(save, key="overlay:splines")),
    etlcdb.overlay_splines,
    etlcdb.splines_image,
    etlcdb.transform_splines,
    bezier.kvg_bbox,
    bezier.kvg_inject,
    etlcdb.overlay_skeleton,
    etlcdb.zhang_skeleton,
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

    # query = """
    #     SELECT id, entry, literal, unicode, groups, data
    #     FROM   glyph
    #     WHERE  entry LIKE 'ETL1/%'
    #     AND    groups = 'KATAKANA'
    #     ORDER  BY literal
    # """

    for glyph in glyph_iterator(query):
        glyph = filters(glyph)
        if glyph["skip"]:
            continue
        pipeline(glyph)
