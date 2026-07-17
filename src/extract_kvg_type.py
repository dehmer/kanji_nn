#!/usr/bin/env python3

import zipfile
import re
import xml.etree.ElementTree as ET
import unicodedata

def stroke_type_name(type_cp):
    """type_cp like 'U+31D6' -> 'CJK STROKE HG'"""
    ch = chr(int(type_cp[2:], 16))
    return unicodedata.name(ch)

if __name__ == '__main__':
    ZIP_PATH = "/Users/dehmer/Public/Data/kanjivg-20250816-all.zip"

    archive = zipfile.ZipFile(ZIP_PATH)
    for entry in archive.namelist():
        if not entry.endswith(".svg"): continue

        match = re.search('^kanji/([0-9a-f]+)([-]*(.*))\\.svg$', entry)
        unicode = match.group(1).upper()

        # There are 88 unique variant types for a total of 4,958 entries.
        variant = match.group(3) if len(match.groups(3)) else None
        if variant:
            continue

        # We strictly stick to 4-digit code points.
        # This will exclude only 7 entries.
        if not unicode.startswith('0'):
            continue

        cp = f'U+{unicode[1:]}' # strip leading 0
        literal = chr(int(unicode, 16))

        # U+5B57
        # if literal != '音':
        #     continue

        # print(cp, literal)

        content = archive.read(entry).decode("utf-8")
        # print(content)
        root = ET.fromstring(content)

        # print(root)

        paths = root.findall(".//{http://www.w3.org/2000/svg}path")

        for i, path in enumerate(paths):
            key = '{http://kanjivg.tagaini.net}type'
            if not key in path.attrib.keys():
                print(f"{literal}\t{i}\t\\N\t\\N\t\\N")
            else:
                kvg_type = path.attrib[key]
                code_point = f"U+{ord(kvg_type[0]):04X}"
                name = stroke_type_name(code_point)
                print(f"{literal}\t{i}\t{kvg_type[0]}\t{code_point}\t{name}")
