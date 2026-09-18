import io
from PIL import Image
from kanji_nn.io import groups

def to_tsv(glyph):
    literal = glyph["literal"]
    image = glyph["image"]
    width, height = image.size

    data = io.BytesIO()
    image.save(data, format='PNG')

    fields = [
        glyph["id"],
        glyph["dataset"],
        literal if literal != "\\" else "\\\\",
        glyph["unicode"],
        ",".join(groups(literal)),
        str(width),
        str(height),
        image.mode,
        "\\\\x" + data.getvalue().hex()
    ]

    print("\t".join(fields))
    return glyph
