from collections import Counter
import numpy as np
from skan.csr import Skeleton
import networkx as nx
from .connected_features import fold


def consolidate_graph(glyph):
    pixel_graph = glyph["pixel_graph"]

    parallel_branches = pixel_graph.parallel_branches()
    if len(parallel_branches):
        print(glyph["literal"], glyph["id"], len(parallel_branches))
        for branch in parallel_branches:
            nodes, branches = branch
            print("  ", branch, pixel_graph.distance(*nodes), pixel_graph.coords[nodes, :])

    return glyph
