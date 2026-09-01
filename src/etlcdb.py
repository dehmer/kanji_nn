#!/usr/bin/env python3
import sys
from signal import signal, SIGINT
from functools import partial, reduce
import numpy as np
from PIL import Image
from scipy.ndimage import label

from kanji_nn.predef import tap
from kanji_nn.etlcdb import glyph_iterator
import kanji_nn.etlcdb as etlcdb
import kanji_nn.bezier as bezier

def fn_name(fn):
    if isinstance(fn, partial):
        return fn_name(fn.func)  # Recursively unwrap in case of nested partials
    return getattr(fn, "__name__", str(fn))


def skippable(fn):
    def inner(glyph):
        if glyph["skip"]:
            return glyph
        else:
            return fn(glyph)
    return inner


def compose(*fns):
    skippable_fns = list(map(skippable, fns))
    return lambda x: reduce(lambda acc, f: f(acc), reversed(skippable_fns), x)


def await_input(glyph):
    input("...")
    return glyph


def terminate(_):
    exit()


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
    # terminate,
    # await_input,
    # partial(save, image_fn=lambda glyph: glyph["image:binary"]),
    partial(save, image_fn=etlcdb.skeleton_overlay),
    # partial(show, image_fn=etlcdb.splines_overlay),
    # partial(show, image_fn=etlcdb.skeleton_overlay),
    # partial(show, image_fn=lambda glyph: glyph["image:splines"]),
    etlcdb.splines_image,
    etlcdb.transform_splines,
    etlcdb.zhang_skeleton,
    # partial(show, image_fn=etlcdb.features_overlay),
    etlcdb.flag_component_count,
    bezier.kvg_bbox,
    bezier.kvg_inject,
    partial(etlcdb.remove_border_noise),
    partial(etlcdb.remove_noise, min_size=5),
    etlcdb.otsu,
    tap(lambda x: print(x["literal"], x["id"])),
    # tap(lambda x: print(x)),
)


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())

    # query = """
    #     SELECT id, dataset, literal, unicode, groups, data
    #     FROM   glyph
    #     WHERE  entry LIKE 'ETL1/%'
    #     AND    groups = 'KATAKANA'
    #     AND    literal = 'イ'
    #     ORDER  BY literal
    # """

    # query = """
    #     SELECT id, dataset, literal, unicode, groups, data
    #     FROM   glyph
    #     WHERE  entry LIKE 'ETL1/%'
    #     AND    groups = 'KATAKANA'
    #     ORDER  BY literal
    # """


    # query = """
    #     SELECT id, dataset, literal, unicode, groups, data
    #     FROM   glyph
    #     WHERE  id in (
    #         '177b9a1a-2878-4628-b433-c6982251024e',
    #     )
    # """

    query = """
        SELECT   id, dataset, literal, unicode, groups, data
        FROM     glyph
        WHERE    literal = 'ア'
        AND      dataset = 'ETL5'
        ORDER BY literal
    """

    for glyph in glyph_iterator(query):
        glyph = pipeline(glyph)
        if glyph["skip"]:
            print(f'skipped: [{glyph["literal"]} - ${glyph["id"]}] - {glyph["reason"]}')
