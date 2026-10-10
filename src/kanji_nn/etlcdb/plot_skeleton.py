import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skan.csr import Skeleton

def _edt_box(junction, edt, fac=2.0):
    radius = edt[tuple(junction.T)]
    extent = fac * radius

    # Matplotlib Anchors: Rectangle takes the bottom-left corner (X, Y).
    # In image coordinates: X = col, Y = row.
    # We shift by 'radius' to center the square over the pixel (c, r).
    bottom_left_x = junction[1] - extent / 2
    bottom_left_y = junction[0] - extent / 2

    # Create the Rectangle patch
    # edgecolor: box border color, facecolor: 'none' leaves it transparent
    return patches.Rectangle(
        (bottom_left_x, bottom_left_y),
        extent,
        extent,
        linewidth=1,
        edgecolor='red',
        facecolor='none'
    )


def plot_skeleton(glyph, image_fn=lambda _: None, edt_boxes=False):
    """
    Plots the raw, fragmented paths extracted by Skan.
    Each distinct topological branch is colored differently.
    """
    size = glyph["size"]
    mask = glyph["skeleton:mask"]
    edt = glyph["skeleton:edt"]
    skeleton = Skeleton(mask)
    coords = skeleton.coordinates

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)

    image = image_fn(glyph)
    if image:
        if hasattr(image, "convert"):
            image = np.array(image.convert("RGB"))
        else:
            image = np.array(image)

        extent = [0, size[0], size[1], 0]
        ax.imshow(image, extent=extent)

    # Loop through every isolated path fragment found by Skan
    for path_index in range(0, skeleton.n_paths):
        path = skeleton.path(path_index)
        x = coords[path, 1]
        y = coords[path, 0]
        ax.plot(x, y, linewidth=3, zorder=0)

    if edt_boxes:
        # Highlight the junctions/endpoints using degrees
        degrees = skeleton.degrees
        junctions = coords[degrees > 2]

        for junction in junctions:
            ax.add_patch(_edt_box(junction, edt, 2.5))

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.show()
    return glyph