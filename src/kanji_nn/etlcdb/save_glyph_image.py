

def save_glyph_image(glyph, image_fn):
    id = glyph["id"]
    image = image_fn(glyph)
    image.save(f"data/images/{id}.png")
    return glyph
