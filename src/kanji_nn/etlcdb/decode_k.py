from PIL import Image
from kanji_nn.etlcdb.unpack import unpack
from kanji_nn.cli import literal_to_hex
from kanji_nn.etlcdb.co59 import co59_to_unicode


_fields = {
    "skip:168": "pad:168",
    "co_59_code": "bits:12",
    "skip:180": "pad:180",
    "image_data": "bytes:2700"
}


def decode_k(chunk: AnyStr) -> dict[str, Any]:
    record = unpack(chunk, _fields)
    code = tuple([b.uint for b in record["co_59_code"].cut(6)])
    literal = co59_to_unicode(code)
    unicode = f"U+{literal_to_hex(literal)}"
    image = Image.frombytes("F", (60, 60), record["image_data"], "bit", 6)
    image = Image.eval(image.convert("L"), lambda x: x * 4)

    return {
        "literal": literal,
        "unicode": unicode,
        "image": image
    }
