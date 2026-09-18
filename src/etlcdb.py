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
import kanji_nn.plot as plot


def await_input(glyph):
    input("...")
    return glyph


def terminate(_):
    exit()


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


pipeline = compose(
    # terminate,
    # await_input,
    # partial(etlcdb.save_glyph_image, image_fn=etlcdb.skeleton_overlay),

    # etlcdb.plot_stroke_assignment,
    # etlcdb.plot_margin_histogram,
    # etlcdb.knn,
    partial(plot.show_pixel_graph, image_fn=lambda g: g["image:binary"]),
    # partial(etlcdb.show_glyph_image, image_fn=lambda g: g["image:binary"]),
    etlcdb.consolidate_graph,
    etlcdb.skeleton_graph,

    # Scale/translate splines to skeleton bounding box.
    etlcdb.resample_splines,
    etlcdb.transform_splines,
    etlcdb.zhang_skeleton,
    # Strict (padding=0): catch fragmentation as a quality signal
    partial(etlcdb.flag_feature_count, padding=0),

    # Bring KanjiVG to the party:
    bezier.kvg_bbox,
    bezier.kvg_inject,
    # Generous (padding=3): cleanup should not fragment real strokes
    partial(etlcdb.remove_noise, min_size=5, margin=2, padding=3),
    etlcdb.otsu,
    etlcdb.flag_label_mismatch,
    # tap(lambda x: print(x["literal"], x["id"])),
)


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())

    # query = """
    #     SELECT id, dataset, literal, unicode, groups, data
    #     FROM   glyph
    #     WHERE  dataset = 'ETL1'
    #     AND    groups = 'KATAKANA'
    #     AND    literal = 'ア'
    #     ORDER  BY literal
    # """

    # query = """
    #     SELECT id, dataset, literal, unicode, groups, data
    #     FROM   glyph
    #     WHERE  entry LIKE 'ETL1/%'
    #     AND    groups = 'KATAKANA'
    #     ORDER  BY literal
    # """


    query = """
        SELECT id, dataset, literal, unicode, groups, data
        FROM   glyph
        WHERE  id in (
            '96479640-3101-4f30-a448-9388263ab408',
            'a46aa4db-8996-4227-9974-443959b5b40e'
        )
    """

    # query = """
    #     SELECT   id, dataset, literal, unicode, groups, data
    #     FROM     glyph
    #     WHERE    literal = '点'
    #     ORDER BY literal
    # """

    # query = """
    #     SELECT id, dataset, literal, unicode, groups, data
    #     FROM   glyph
    #     WHERE  dataset = 'ETL9G'
    #     AND    groups LIKE '%KANJI%'
    # """

    total = 0
    rejected = 0
    for glyph in glyph_iterator(query):
        total += 1
        glyph = pipeline(glyph)
        if glyph["skip"]:
            rejected += 1
            # print(f'skipped: [{glyph["literal"]} - {glyph["id"]}] - {glyph["reason"]}')
            # if "image:binary" in glyph:
            #     etlcdb.save_glyph_image(glyph, image_fn=lambda glyph: glyph["image:binary"])
            # else:
            #     etlcdb.save_glyph_image(glyph, image_fn=lambda glyph: glyph["image"])

    print("total", total)
    print("rejected", rejected)
    print("percent", (rejected / total) * 100)