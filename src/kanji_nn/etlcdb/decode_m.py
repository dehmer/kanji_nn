from PIL import Image
import jaconv
from kanji_nn.etlcdb.unpack import unpack
from kanji_nn.cli import literal_to_hex


# _fields = {
#     "skip:32": "pad:48",
#     "jis_x_0201": "hex:8",
#     "skip:200": "pad:200",
#     "image_data": "bytes:2016"
# }

_fields = {
    "skip:16(1)": "pad:16",
    "code": "bytes:2",
    "skip:16(2)": "pad:16",
    "jis_x_0201": "hex:8",
    "skip:200": "pad:200",
    "image_data": "bytes:2016"
}


def decode_m(dataset, chunk: AnyStr) -> dict[str, Any]:
    record = unpack(chunk, _fields)

    # TODO: can this code help in any way?
    code = record["code"].decode('ascii')

    if record["jis_x_0201"] == "00":
        return {"skip": True}

    literal = bytes.fromhex(record["jis_x_0201"]).decode("shift_jis")
    literal = jaconv.h2z(literal, digit=False, ascii=False)

    # TODO: do we need replacements for ETL7
    # TODO: dakuten, handakuten handling
    if dataset == 'ETL7':
        literal = jaconv.kata2hira(literal)
    else:
        literal = literal.replace("ィ", "ヰ").replace("ェ", "ヱ")

    unicode = f"U+{literal_to_hex(literal)}"
    image = Image.frombytes("F", (64, 63), record["image_data"], "bit", 4)
    image = Image.eval(image.convert("L"), lambda x: x * 17)

    return {
        "literal": literal,
        "unicode": unicode,
        "image": image
    }
