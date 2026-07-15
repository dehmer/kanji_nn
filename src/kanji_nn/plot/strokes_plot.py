import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def _strokes(strokes, title, figsize):
    figure = plt.figure(figsize=figsize)
    for i, stroke in enumerate(strokes):
        xy = stroke[:, (0, 1)]
        plt.plot(xy[:, 0], xy[:, 1], color='black', zorder=1)
        plt.scatter(xy[:, 0],  xy[:, 1], marker='o', color='red', zorder=2, alpha=.3)
        plt.scatter(xy[0, 0],  xy[0, 1], marker='o', color='green', zorder=3)
        plt.scatter(xy[-1, 0], xy[-1, 1], marker='o', color='blue', zorder=3)

    # Kanji coordinate systems usually start at the top-left (invert Y-axis)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_xlim([0, 1]) # assume fixed x values [0, 1]
    ax.set_ylim([1, 0]) # assume fixed y values [0, 1]
    ax.set_axis_off()

    if title:
        ax.set_title(title)

    return figure


def save(filename, strokes, title=None, figsize=(6, 6)):
    figure = _strokes(strokes, title, figsize)
    plt.savefig(filename)
    plt.close(figure)


def show(strokes, title=None, figsize=(6, 6)):
    figure = _strokes(strokes, title, figsize)
    plt.show()
