import zipfile
import re
import numpy as np
import xml.etree.ElementTree as ET
from svg.path import Path, parse_path

from . import code_point
from .. import vg

def _extract_paths(content: str) -> Path:
    """
    Extract and parse all paths in a given SVG string.
    """

    tag = "path"
    namespace = "{http://www.w3.org/2000/svg}"
    xpath = ".//" + namespace + tag
    root = ET.fromstring(content)
    return [parse_path(path.attrib['d']) for path in root.findall(xpath)]


def _parse_entry(archive, entry):
    content = archive.read(entry).decode("utf-8")
    return _extract_paths(content)


def parse_archive(filename, filter):
    archive = zipfile.ZipFile(filename)

    for entry in archive.namelist():
        if not entry.endswith(".svg"): continue

        match = re.search('^kanji/([0-9a-f]+)([-]*(.*))\\.svg$', entry)
        unicode = match.group(1).upper()

        # We strictly stick to 4-digit code points.
        # This will exclude only 7 entries.
        if not unicode.startswith('0'):
            continue

        cp = f'U+{unicode[1:]}' # strip leading 0
        literal = chr(int(unicode, 16))

        # There are 88 unique variant types for a total of 4,958 entries.
        variant = match.group(3) if len(match.groups(3)) else None

        groups = code_point.groups(literal)

        if variant:
            groups.append(f"VARIANT:{variant}")

        if not filter(literal, groups):
            continue

        paths = _parse_entry(archive, entry)
        yield cp, paths
