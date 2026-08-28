import io
import psycopg
from psycopg.rows import dict_row
from PIL import Image
import uuid

from kanji_nn.data import conninfo_from_env


_conninfo = conninfo_from_env("kanji_nn")
_conn = psycopg.connect(_conninfo)


def _dict_cursor(name):
    return _conn.cursor(name, row_factory=dict_row)


def _process(row):
    image = Image.open(io.BytesIO(row["data"]))
    image.load()
    glyph = {k: v for k, v in row.items() if k != "data"}
    return glyph | {"image": image}


def glyph_iterator(query):
    cursor_name = f"gi_{uuid.uuid4().hex}"
    with _dict_cursor(name=cursor_name) as cur:
        cur.execute(query)
        for row in cur:
            yield _process(row)
