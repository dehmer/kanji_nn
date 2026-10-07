import numpy as np
import matplotlib.pyplot as plt
from skan.csr import Skeleton

def plot_skeleton(glyph, image_fn=lambda _: None):
    """
    Plots the raw, fragmented paths extracted by Skan.
    Each distinct topological branch is colored differently.
    """
    size = glyph["size"]
    mask = glyph["skeleton:mask"]
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

    # Highlight the junctions/endpoints using degrees
    degrees = skeleton.degrees
    junctions = coords[degrees > 2]
    endpoints = coords[degrees == 1]

    if len(junctions) > 0:
        ax.scatter(junctions[:, 1], junctions[:, 0], color='red', facecolors='none', s=120, zorder=1)

    if len(endpoints) > 0:
        ax.scatter(endpoints[:, 1], endpoints[:, 0], color='red', alpha=0.4, marker='o', s=60, zorder=1)

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.show()
    return glyph