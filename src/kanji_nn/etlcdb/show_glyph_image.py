

def show_glyph_image(glyph, image_fn):
    image = image_fn(glyph)
    image.show()
    return glyph
