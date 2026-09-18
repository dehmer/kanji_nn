from collections import Counter
import numpy as np
from skan.csr import Skeleton
import networkx as nx
from .connected_features import fold
from .pixel_graph import PixelGraph


def consolidate_graph(glyph):
    g: PixelGraph = glyph["pixel_graph"]

    clusters = g.junction_clusters()
    clusters = [c for c in clusters if len(c) > 1]
    for c in clusters:
      print(glyph["id"], c, g.coords[c])


    return glyph
