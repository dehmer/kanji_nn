import numpy as np
import kanji_nn.etlcdb.skeleton_ops as sops


def consolidate_skeleton(glyph):

    glyph = sops.prune_parallel_paths(glyph)
    glyph = sops.dissolve_t_junctions(glyph)

    return glyph
