#!/usr/bin/env python3
import os
import sys
from signal import signal, SIGINT
import pathlib
import zipfile
import uuid
import json

from kanji_nn.predef import compose, tap
from kanji_nn.etlcdb import decode_b9, decode_g9
from kanji_nn.etlcdb import decode_b8, decode_g8
from kanji_nn.etlcdb import decode_k, decode_m, decode_c
from kanji_nn.io import groups

# Load pre-allocated UUIDs:
ids = {}
with open("data/glyphs_ids.json", 'r') as file:
    ids = json.load(file)


pipeline = compose(
    # TODO: do something useful
    tap(lambda x: print(x))
)


def parse_entry(archive, decoder, entry, filter):
    byte_length, decode = decoder
    with archive.open(entry, "r") as f:
        offset = 0

        # skip first
        if entry.split("/")[0] in ["ETL8B", "ETL9B"]:
            f.read(byte_length)
            offset = byte_length

        while chunk := f.read(byte_length):
            key = f"{entry}:{offset}"
            glyph = {"offset": offset, "entry": entry, "id": ids[key]} | decode(chunk)

            if filter(glyph["literal"]):
                pipeline(glyph)
            offset += byte_length


def parse_archive(filename, decoder, filter):
    archive = zipfile.ZipFile(filename)
    for entry in archive.namelist():
        if entry.endswith("INFO"): continue
        parse_entry(archive, decoder, entry, filter)


if __name__ == "__main__":
    signal(SIGINT, lambda _, __: sys.exit())

    PATH = "data/etlcdb"

    decoders = {
        "m":  (2052, decode_m),
        "k":  (2745, decode_k),
        "c":  (2952, decode_c),
        "b8": ( 512, decode_b8),
        "g8": (8199, decode_g8),
        "b9": ( 576, decode_b9),
        "g9": (8199, decode_g9),
    }

    datasets = {
        "ETL1":  decoders["m"],  # katakana
        "ETL2":  decoders["k"],  # font-rendered
        "ETL3":  decoders["c"],  # numerals, uppercase alphabets, symbols
        "ETL4":  decoders["c"],  # hiragana
        "ETL5":  decoders["c"],  # katakanas
        "ETL6":  decoders["m"],  # katakanas, numerals, uppercase alphabets, symbols
        "ETL7":  decoders["m"],  # hiragana, dakuten, handakuten
        "ETL8B": decoders["b8"], # hiragana, kanji
        "ETL8G": decoders["g8"], # hiragana, kanji
        "ETL9B": decoders["b9"], # hiragana, kanji
        "ETL9G": decoders["g9"], # kanji only
    }

    filter = lambda literal: True
    # filter = lambda literal: groups(literal) == ["KATAKANA"]

    for (dirpath, dirnames, filenames) in os.walk(PATH):
        filenames.sort()
        for filename in filenames:
            if not filename.endswith("zip"): continue
            path = pathlib.Path(f"{dirpath}/{filename}")
            if not path.stem in datasets.keys(): continue
            parse_archive(path, datasets[path.stem], filter)

    # with open("glyphs_ids.json", "w") as file:
    #     file.write(json.dumps(ids)) # use `json.loads` to do the reverse