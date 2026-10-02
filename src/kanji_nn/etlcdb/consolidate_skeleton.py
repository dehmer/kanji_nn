import numpy as np
import kanji_nn.etlcdb.skeleton_ops as sops


def consolidate_skeleton(glyph):
    degrees = glyph["skeleton:degrees"]
    xy = glyph["skeleton:xy"]
    inner_edt = np.median(sops.edt(glyph)[degrees == 2])
    radius = inner_edt + 2 * np.sqrt(inner_edt) + 1e-5  # fixed once, no drift

    while True:
        clusters = sops.junction_clusters(glyph, radius)
        clusters = sops.cluster_paths(glyph, clusters)
        match = next((
            c for c in clusters
            if len(c["junctions"]) == 2
            and len(c["incoming"]) == 1
            and len(c["outgoing"]) == 1
        ), None)

        if match is None:
            break

        glyph = sops.merge_through(glyph, match)


    # 7 clusters remaining
    clusters = sops.junction_clusters(glyph, radius)
    clusters = sops.cluster_paths(glyph, clusters)
    for cluster in clusters:
        print(cluster)
        print(xy[cluster["junctions"], :])

    return glyph
