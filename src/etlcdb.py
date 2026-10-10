#!/usr/bin/env python3
import sys
from signal import signal, SIGINT
from functools import partial, reduce
import numpy as np
from PIL import Image
from scipy.ndimage import label
from pathlib import Path

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

# image accessor/creation function:
image = lambda glyph: glyph["image"]
binary_image = lambda glyph: glyph["image:binary"]
skeleton_image = lambda glyph: glyph["image:skeleton"]

# image filename functions:
image_filename = lambda glyph: f"data/images/{glyph['id']}.png"
binary_image_filename = lambda glyph: f"data/images/{glyph['id']}-binary.png"
skeleton_image_filename = lambda glyph: f"data/images/{glyph['id']}-skeleton.png"


pipeline = compose(
    # partial(etlcdb.save_glyph_image, image_fn=skeleton_image, filename_fn=skeleton_image_filename),
    # partial(etlcdb.save_glyph_image, image_fn=binary_image, filename_fn=binary_image_filename),
    # partial(etlcdb.save_glyph_image, image_fn=image, filename_fn=image_filename),
    # partial(etlcdb.plot_glyph_image, image_fn=image),

    etlcdb.plot_stroke_assignments,
    etlcdb.skeleton_assignment,

    # partial(etlcdb.plot_t_junctions),
    # partial(etlcdb.plot_skeleton, image_fn=image),
    # etlcdb.consolidate_skeleton,

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

    # Remove detected noise in original and binary image.
    etlcdb.remove_noise,

    # Generous (padding=3): cleanup should not fragmentize real strokes.
    partial(etlcdb.detect_noise, min_size=5, margin=2, padding=3),
    etlcdb.otsu,
    etlcdb.flag_label_mismatch,
    tap(lambda x: print(x["literal"], x["id"])),
)


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())

    # Load query:
    script = "single_id.sql"
    MODULE_DIR = Path(__file__).resolve().parent.parent
    file_path = MODULE_DIR / "queries" / script
    with open(file_path, "r", encoding="utf-8") as file:
        query = file.read()

    total = 0
    rejected = 0
    for glyph in glyph_iterator(query):
        total += 1
        glyph = pipeline(glyph)
        if glyph["skip"]:
            rejected += 1
            print(f'skipped: [{glyph["literal"]} - {glyph["id"]}] - {glyph["reason"]}')
            if "image:binary" in glyph:
                etlcdb.save_glyph_image(glyph, image_fn=lambda glyph: glyph["image:binary"])
            else:
                etlcdb.save_glyph_image(glyph, image_fn=lambda glyph: glyph["image"])

    print("total", total)
    print("rejected", rejected)
    print("percent", (rejected / total) * 100)