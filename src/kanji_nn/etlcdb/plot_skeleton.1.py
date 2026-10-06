import numpy as np
import matplotlib.pyplot as plt

def plot_skeleton(glyph):
    """
    Plots the raw, fragmented paths extracted by Skan.
    Each distinct topological branch is colored differently.
    """
    size = glyph["size"]
    xy = glyph["skeleton:xy"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)

    # Loop through every isolated path fragment found by Skan
    for path in glyph["skeleton:paths"]:
        x = xy[path, 0]
        y = xy[path, 1]
        ax.plot(x, y, linewidth=3, zorder=0)

    # Highlight the junctions/endpoints using degrees
    degrees = glyph["skeleton:degrees"]
    junctions = xy[degrees > 2]
    endpoints = xy[degrees == 1]

    if len(junctions) > 0:
        ax.scatter(junctions[:, 0], junctions[:, 1], color='red', facecolors='none', s=120, zorder=1)

    if len(endpoints) > 0:
        ax.scatter(endpoints[:, 0], endpoints[:, 1], color='red', alpha=0.4, marker='o', s=60, zorder=1)

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.show()
    return glyph