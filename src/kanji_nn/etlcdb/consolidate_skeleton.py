import numpy as np
import kanji_nn.etlcdb.skeleton_ops as sops


def consolidate_skeleton(glyph):
    path_lengths = glyph["skeleton:lengths"]
    xy = glyph["skeleton:xy"]

    clusters = sops.junction_clusters(glyph)
    valence = sops.cluster_paths(glyph, clusters)

    for cluster in valence:
        if len(cluster["junctions"]) < 2:
            continue

        print(cluster)

    return glyph