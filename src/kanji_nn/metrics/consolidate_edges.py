import heapq
import numpy as np
from collections import OrderedDict
from itertools import groupby


def consolidate_edges(stroke):
    t = stroke.t
    edges = stroke.props.get('edges', list())

    regions = []
    region_duration = 0
    for i in range(0, len(edges), 2):
        start = edges[i][1]
        end = edges[i + 1][1]
        duration = t[edges[i + 1][1]] - t[edges[i][1]]
        region_duration += duration
        regions.append((start, end, duration))

    for region in regions:
        print(region)

    duration = stroke.duration
    print('duration', duration)
    print('region_duration', region_duration, region_duration / duration)
    return stroke
