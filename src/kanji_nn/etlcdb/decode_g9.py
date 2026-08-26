from PIL import Image
from kanji_nn.etlcdb.unpack import unpack
from kanji_nn.cli import literal_to_hex


_fields = {
    "skip:16": "pad:16",
    "jis_x_0208": "hex:16",
    "skip:480": "pad:480",
    "image_data": "bytes:8128"
}


def decode_g9(chunk: AnyStr) -> dict[str, Any]:
    record = unpack(chunk, _fields)
    code = "1b2442" + record["jis_x_0208"] + "1b2842"
    literal = bytes.fromhex(code).decode("iso2022_jp")
    unicode = f"U+{literal_to_hex(literal)}"
    image = Image.frombytes("F", (128, 127), record["image_data"], "bit", 4)
    image = Image.eval(image.convert("L"), lambda x: x * 17) # 255 / 15 = 17

    return {
        "literal": literal,
        "unicode": unicode,
        "image": image
    }
