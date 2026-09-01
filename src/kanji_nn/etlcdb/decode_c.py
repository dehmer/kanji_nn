from PIL import Image
import jaconv
from kanji_nn.etlcdb.unpack import unpack
from kanji_nn.cli import literal_to_hex


_T56 = '0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);\'|/STUVWXYZ ,%="!'
def t56(c):
    return _T56[c]


_fields = {
    "skip:72": "pad:72",
    "jis_x_0201": "hex:8",
    "skip:28": "pad:28", # remainder of JIS X 0201
    "skip:36": "pad:36", # EBCDIC Code
    "t56_4cc": "bits:24", # T56 4 character code
    "skip:12": "pad:12", # spaces
    "skip:1548": "pad:1548",
    "image_data": "bytes:2736"
}


def convert_literal(t56_code, literal):

    # hiragana:
    if t56_code[0] == "H":
        literal = jaconv.h2z(literal)
        literal = jaconv.kata2hira(literal)
        if t56_code[2:] == "WI":
            literal = "ゐ"
        elif t56_code[2:] == "WE":
            literal = "ゑ"
    elif t56_code[0] == "K":
        literal = jaconv.h2z(literal)
        if t56_code[2:] == "WI":
            literal = "ヰ"
        elif t56_code[2:] == "WE":
            literal = "ヱ"

    return literal


def decode_c(dataset, chunk: AnyStr) -> dict[str, Any]:
    record = unpack(chunk, _fields)
    t56_code = ''.join([t56(b.uint) for b in record["t56_4cc"].cut(6)])

    literal = bytes.fromhex(record["jis_x_0201"]).decode("shift_jis")
    literal = convert_literal(t56_code, literal)
    unicode = f"U+{literal_to_hex(literal)}"

    image = Image.frombytes('F', (72,76), record["image_data"], 'bit', 4)
    image = Image.eval(image.convert("L"), lambda x: x * 17)

    return {
        "literal": literal,
        "unicode": unicode,
        "image": image
    }
