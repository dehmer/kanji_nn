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


def terminate(_):
    exit()


def splines_overlay(glyph):
    base = glyph["image:binary"].convert("RGB")
    mask = glyph["image:splines"].convert("L")
    green = Image.new("RGB", base.size, (0, 255, 0))
    return Image.composite(green, base, mask)


def skeleton_overlay(glyph):
    base = glyph["image:binary"].convert("RGB")
    mask = glyph["image:skeleton"] # mode "L", 0/255
    red = Image.new("RGB", base.size, (255, 0, 0))
    return Image.composite(red, base, mask)


def show(glyph, image_fn):
    image = image_fn(glyph)
    image.show()
    return glyph


def save(glyph, image_fn):
    id = glyph["id"]
    image = image_fn(glyph)
    image.save(f"data/images/{id}.png")
    return glyph


pipeline = compose(
    await_input,
    terminate,
    partial(show, image_fn=splines_overlay),
    partial(show, image_fn=skeleton_overlay),
    partial(show, image_fn=lambda glyph: glyph["image:splines"]),
    etlcdb.splines_image,
    etlcdb.transform_splines,
    bezier.kvg_bbox,
    bezier.kvg_inject,
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
