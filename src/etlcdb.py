#!/usr/bin/env python3
import sys
from signal import signal, SIGINT

from kanji_nn.predef import compose, tap
import kanji_nn.etlcdb as etlcdb
from kanji_nn.etlcdb import glyph_iterator


def show(glyph):
    image = glyph["image:overlay"]
    image.show()
    input("...")


pipeline = compose(
    tap(show),
    etlcdb.overlay_skeleton,
    etlcdb.zhang_skeleton,
    etlcdb.otsu,
    tap(lambda x: print(x))
)


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())

    query = """
        SELECT id, entry, literal, unicode, groups, data
        FROM   glyph
        WHERE  entry LIKE 'ETL1/%'
        AND    groups = 'KATAKANA'
        ORDER  BY literal
    """

    for glyph in glyph_iterator(query):
        pipeline(glyph)
