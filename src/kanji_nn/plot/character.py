import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def _strokes(strokes, figsize):
    column_count = strokes.shape[1]
    strokes = strokes if column_count == 3 else strokes[:,[1, 2, 4]]

    # For n strokes lookup n-1 split indices.
    # Split happens at index, so add 1 to each index.
    split_indices = np.where(strokes[:, 2] == 0)[0] + 1
    lines = np.split(strokes, split_indices[:-1]) # use n-1 indices, drop last

    figure = plt.figure(figsize=figsize)
    for i, stroke in enumerate(lines):
        stroke = np.array(stroke)
        plt.plot(stroke[:, 0], stroke[:, 1], color='black', zorder=1)
        plt.scatter(stroke[:, 0],  stroke[:, 1], marker='o', color='red', zorder=2, alpha=.3)
        plt.scatter(stroke[0, 0],  stroke[0, 1], marker='o', color='green', zorder=3)
        plt.scatter(stroke[-1, 0], stroke[-1, 1], marker='o', color='blue', zorder=3)

    # Kanji coordinate systems usually start at the top-left (invert Y-axis)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_xlim([0, 1]) # assume fixed x values [0, 1]
    ax.set_ylim([1, 0]) # assume fixed y values [0, 1]
    ax.set_axis_off()

    return figure


def save(filename, strokes, figsize=(6, 6)):
    figure = _strokes(strokes, figsize)
    plt.savefig(filename)
    plt.close(figure)


def show(strokes, figsize=(6, 6)):
    figure = _strokes(strokes, figsize)
    plt.show()
