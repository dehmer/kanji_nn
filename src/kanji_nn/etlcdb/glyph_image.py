import numpy as np
import matplotlib.pyplot as plt


image_fn = lambda glyph: glyph["image"]
filename_fn = lambda glyph: f"data/images/{glyph['id']}.png"


def glyph_image_figure(glyph, image_fn=image_fn, figsize=(9, 9)):
    size = glyph["size"]
    image = image_fn(glyph)

    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))
    else:
        image = np.array(image)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)

    extent = [0, size[0], size[1], 0]
    ax.imshow(image, extent=extent)
    return fig


def plot_glyph_image(glyph, image_fn=image_fn, figsize=(9, 9)):
    fig = glyph_image_figure(glyph=glyph, image_fn=image_fn, figsize=figsize)
    plt.show()
    plt.close(fig)
    return glyph


def save_glyph_image(glyph, image_fn=image_fn, figsize=(9, 9), filename_fn=filename_fn):
    fig = glyph_image_figure(glyph=glyph, image_fn=image_fn, figsize=figsize)
    plt.savefig(filename_fn(glyph))
    plt.close(fig)
    return glyph
