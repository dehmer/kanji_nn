import matplotlib.pyplot as plt
from skan.csr import Skeleton
import kanji_nn.etlcdb.skeleton_ops as sops


def plot_t_junctions(glyph):
    size = glyph["size"]
    edt = glyph["skeleton:edt"]
    mask = glyph["skeleton:mask"]
    skeleton = Skeleton(mask)
    coords = skeleton.coordinates

    nrows = 2
    triplets = sops.t_junctions(glyph)
    num_junctions = len(triplets)
    ncols = (num_junctions + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ((num_junctions + 1) // 2), 8))
    axes = axes.ravel()

    for idx, triplet in enumerate(triplets.items()):
        junction, paths = triplet
        ax = axes[idx]
        ax.set_xlim(0, size[0])
        ax.set_ylim(size[1], 0)
        ax.set_title(f"junction {junction} @ {coords[junction]}, tol=1.0")

        for i, path in enumerate(paths):
            x = coords[path, 1]
            y = coords[path, 0]
            ax.plot(x, y, linewidth=3, zorder=0)

    plt.show()
    plt.close(fig)
    return glyph
