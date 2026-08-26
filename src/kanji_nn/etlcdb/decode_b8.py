from PIL import Image
from kanji_nn.etlcdb.unpack import unpack
from kanji_nn.cli import literal_to_hex


_fields = {
    "skip:16": "pad:16",
    "jis_x_0208": "hex:16",
    "skip:32": "pad:32",
    "image_data": "bytes:504"
}


def decode_b8(chunk: AnyStr) -> dict[str, Any]:
    record = unpack(chunk, _fields)
    code = "1b2442" + record["jis_x_0208"] + "1b2842"
    literal = bytes.fromhex(code).decode("iso2022_jp")
    unicode = f"U+{literal_to_hex(literal)}"
    image = Image.frombytes("1", (64, 63), record["image_data"], "raw")

    return {
        "literal": literal,
        "unicode": unicode,
        "image": image
    }
