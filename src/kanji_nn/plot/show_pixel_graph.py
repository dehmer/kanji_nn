import numpy as np
import matplotlib.pyplot as plt


def show_pixel_graph(glyph, image_fn=None, figsize=(9, 9)):
    pixel_graph = glyph["pixel_graph"]
    size = glyph["size"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)

    # show background image if any:
    if image_fn:
        image = image_fn(glyph)
        extent = [0, size[0], size[1], 0]
        ax.imshow(image, extent=extent, zorder=-1, alpha=0.1)

    # Loop through every isolated path fragment found by Skan
    for branch in pixel_graph.branches:
        coords = pixel_graph.coords[branch]
        ax.plot(coords[:, 0], coords[:, 1], linewidth=2)
        ax.scatter(coords[:, 0], coords[:, 1], s=15, zorder=3)

    # Highlight the junctions/endpoints

    junctions = pixel_graph.junctions()
    ax.scatter(
        pixel_graph.coords[junctions, 0],
        pixel_graph.coords[junctions, 1],
        color='gray', marker='X', s=100, zorder=4, alpha=0.4
    )

    endpoints = pixel_graph.endpoints()
    ax.scatter(
        pixel_graph.coords[endpoints, 0],
        pixel_graph.coords[endpoints, 1],
        color='gray', marker='o', s=60, zorder=4, alpha=0.4
    )

    # Formatting for traditional image coordinate space (Y-axis inverted)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.title(f"Pixel Graph: {glyph["literal"]}\n{glyph["id"]}", fontsize=16)
    plt.show()
    return glyph
