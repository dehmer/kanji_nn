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


image = lambda glyph: glyph["image"]
binary_image = lambda glyph: glyph["image:binary"]
skeleton_image = lambda glyph: glyph["image:skeleton"]
image_filename = lambda glyph: f"data/images/{glyph['id']}.png"
binary_image_filename = lambda glyph: f"data/images/{glyph['id']}-binary.png"
skeleton_image_filename = lambda glyph: f"data/images/{glyph['id']}-skeleton.png"

pipeline = compose(
    # partial(etlcdb.save_glyph_image, image_fn=skeleton_image, filename_fn=skeleton_image_filename),
    # partial(etlcdb.save_glyph_image, image_fn=binary_image, filename_fn=binary_image_filename),
    # partial(etlcdb.save_glyph_image, image_fn=image, filename_fn=image_filename),
    # partial(etlcdb.plot_glyph_image, image_fn=image),

    # etlcdb.plot_stroke_assignments,
    # etlcdb.skeleton_assignment,
    partial(etlcdb.plot_t_junctions),
    partial(etlcdb.plot_skeleton, image_fn=image),
    etlcdb.consolidate_skeleton,
    # partial(etlcdb.plot_skeleton, image_fn=skeleton_image_from_mask),

    # Parametric curves -> euclidean space:
    etlcdb.resample_splines,

    # Scale/translate splines to skeleton bounding box.
    etlcdb.transform_splines,
    etlcdb.skeleton_graph,
    etlcdb.zhang_skeleton,

    # Strict (padding=0): catch fragmentation as a quality signal
    partial(etlcdb.flag_feature_count, padding=0),

    # Bring KanjiVG to the party:
    bezier.kvg_bbox,
    bezier.kvg_inject,
    # Generous (padding=3): cleanup should not fragmentize real strokes.
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


    # query = """
    #     SELECT id, dataset, literal, unicode, groups, data
    #     FROM   glyph
    #     WHERE  id in (
    #         'a4be61d7-72e6-47ca-ab5a-3bc8f2a5d089',
    #         '33c35fa2-25b8-4343-9482-4a1bca556119',
    #         '1faba01a-01bd-4de6-98ce-fa5ae00b602a',
    #         'bb2db137-9c9b-4b45-9fb3-894320006cce',
    #         '5a68c7bc-756c-468d-9956-e304b297dfc9',
    #         '11b3a4f8-bb22-40a9-bd3f-728b78c67007',
    #         '6c57a1a1-76f4-4a2a-9181-40fa905b790f',
    #         '1aa5301d-6ecc-4c76-b2de-ef970671fd18',
    #         '5fe45926-92c5-4fc6-9c3a-d2ccba84425a',
    #         '402e0c73-7d8a-4198-a3a7-08784e737992',
    #         'ae1fb6a9-524b-4bcb-835c-949cdbb3acf9',
    #         '9df1aa51-81ad-495a-afc5-e6458f410fec',
    #         '5410861c-255c-4210-9a04-c27cbce65539',
    #         '8023983a-8fac-42e7-8c13-9206525a03be',
    #         '59346591-91cb-4f9c-aa46-677513dafd60',
    #         '0e44b49d-8e2c-44ac-925c-a8bf3c7ba32b',
    #         '79c0bf17-3d82-42d9-afd0-9f1ea798cd76',
    #         'd063c157-4da2-4d75-8a59-ed5947fdc316',
    #         'e30a4ae3-4b63-4596-84de-ae5f23710ba7',
    #         'e3f5ced6-65ef-49ea-a0ec-85d2c7d30117',
    #         'c064e05b-98c6-4e95-ac94-fa83ecd19e83',
    #         'bc43a10e-fc60-4b4c-8f4d-c2881194c826',
    #         '26c698ea-fdc9-4c5c-80c1-4db4b4f78742',
    #         'c23acd15-11d1-4636-bf24-2c8a11ef5282',
    #         '4f1b9831-f6f1-47fe-b49b-b74a21283f3c',
    #         'fe4ae151-69eb-442d-bdab-c820d0c95e61',
    #         '6f175a81-738d-4eeb-8e2f-82ebc4bceeed',
    #         '738e0295-6cb4-4236-a90b-333814f0513f',
    #         '9aaf9d39-ab03-4ae4-adfc-29958bcd5a0f'
    #     )
    # """

    query = """
        SELECT id, dataset, literal, unicode, groups, data
        FROM   glyph
        WHERE  id in (
            '4f1b9831-f6f1-47fe-b49b-b74a21283f3c'
        )
    """

    # query = """
    #     SELECT   id, dataset, literal, unicode, groups, data
    #     FROM     glyph
    #     WHERE    literal = 'ア'
    #     AND      mode = 'L'
    #     ORDER BY literal
    # """

    # 別
    # query = """
    #     SELECT id, dataset, literal, unicode, groups, data
    #     FROM   glyph
    #     WHERE  dataset = 'ETL9G'
    #     AND    literal = '別'
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