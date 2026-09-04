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


pipeline = compose(
    partial(etlcdb.save_glyph_image, image_fn=etlcdb.skeleton_overlay),
    etlcdb.splines_image,
    etlcdb.transform_splines,
    etlcdb.zhang_skeleton,
    # strict (padding=0): catch fragmentation as a quality signal
    partial(etlcdb.flag_feature_count, padding=0),
    bezier.kvg_bbox,
    bezier.kvg_inject,
    # generous (padding=3): cleanup should not fragment real strokes
    partial(etlcdb.remove_noise, min_size=5, margin=2, padding=3),
    etlcdb.otsu,
    etlcdb.flag_label_mismatch,
    tap(lambda x: print(x["literal"], x["id"])),
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
    #         'e641b5cd-f2d2-44e3-86b0-6e72fe2a65b8'
    #     )
    # """

    query = """
        SELECT   id, dataset, literal, unicode, groups, data
        FROM     glyph
        WHERE    literal = '来'
        ORDER BY literal
    """

    # query = """
    #     SELECT   id, dataset, literal, unicode, groups, data
    #     FROM     glyph
    #     WHERE    groups NOT IN ('DIGIT', 'ROMAJI', 'PUNCTUATION', 'SYMBOL', 'OTHER')
    #     AND      dataset = 'ETL2'
    #     AND      literal IS NOT NULL
    # """

    total = 0
    rejected = 0
    for glyph in glyph_iterator(query):
        total += 1
        glyph = pipeline(glyph)
        if glyph["skip"]:
            rejected += 1
            print(f'skipped: [{glyph["literal"]} - {glyph["id"]}] - {glyph["reason"]}')
            # if "image:binary" in glyph:
            #     etlcdb.save_glyph_image(glyph, image_fn=lambda glyph: glyph["image:binary"])
            # else:
            #     etlcdb.save_glyph_image(glyph, image_fn=lambda glyph: glyph["image"])

    print("total", total)
    print("rejected", rejected)
    print("percent", (rejected / total) * 100)