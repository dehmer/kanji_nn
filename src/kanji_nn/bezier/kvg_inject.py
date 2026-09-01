from kanji_nn.io.KanjiVG import KanjiVG


def kvg_inject(glyph):
    unicode = glyph["unicode"]
    kvg = KanjiVG(unicode)
    paths = kvg.paths
    return glyph | {"kvg:paths": kvg.paths, "num_strokes": len(paths)}
