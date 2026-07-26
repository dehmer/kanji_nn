import xml.etree.ElementTree as ET
from svg.path import Path, parse_path
import unicodedata


DATA_DIR = "data/kanjivg"


class KanjiVG:
    def __init__(self, code_point: str):
        self.code_point = code_point
        file_name = f"{DATA_DIR}/0{code_point[2:].lower()}.svg"
        with open(file_name, mode="r", encoding="utf-8") as file:
            svg_content = file.read()
            self.root = ET.fromstring(svg_content)


    @property
    def _path_elements(self):
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        return self.root.findall('.//svg:path', ns)


    @property
    def paths(self):
        """
        Extract and parse path data (d).
        Natural stroke order is preserved.
        """
        parse = lambda path: parse_path(path.attrib['d'])
        return [parse(p) for p in self._path_elements]


    @property
    def types(self):
        key = '{http://kanjivg.tagaini.net}type'
        info = lambda ch: (ch, unicodedata.name(ch).split(" ")[2])
        return [
            info(p.attrib[key][0]) if key in p.attrib else None
            for p in self._path_elements
        ]
