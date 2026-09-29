from collections import defaultdict


def paths(skeleton):
    """Collect all paths, i.e. pixel indices."""
    return [skeleton.path(i) for i in range(0, skeleton.n_paths)]


def pixel_path_lookup(skeleton):
    """Return reverse lookup: {pixel index: [path index]}."""
    pixel_branches = defaultdict(list)
    for branch_idx, pixel_indices in enumerate(paths(skeleton)):
        for pixel_idx in pixel_indices:
            pixel_branches[int(pixel_idx)].append(branch_idx)

    return pixel_branches
