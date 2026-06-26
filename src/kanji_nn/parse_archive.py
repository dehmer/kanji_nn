import zipfile
import re
import numpy as np

from . import code_point
from . import vg

def parse_archive(filename, filter, n_points):
    archive = zipfile.ZipFile(filename)

    labels = []
    all_strokes = []

    for entry in archive.namelist():
        if not entry.endswith(".svg"): continue

        match = re.search('^kanji/([0-9a-f]+)([-]*(.*))\\.svg$', entry)
        unicode = match.group(1).upper()
        variant = match.group(3) if len(match.groups(3)) else None

        cp = f'U+{unicode}'

        # There are 88 unique variant types; ignore for now:
        # if variant: continue

        literal = chr(int(unicode, 16))
        groups = code_point.groups(literal)

        if variant: groups.append(f"VARIANT:{variant}")
        if not filter(literal, groups): continue

        # strokes = vg.parse_entry_static(archive, entry, 16)
        strokes = vg.parse_entry_to_fixed(archive, entry, n_points)
        all_strokes.append(strokes)
        labels.append(cp)
        # print(cp, literal, groups)

    return all_strokes, labels
