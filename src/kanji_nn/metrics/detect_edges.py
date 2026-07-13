import heapq
import numpy as np

def detect_edges(stroke, signal_key, threshold=0.9):
    t = stroke.t - stroke.t[0]
    signal = stroke.features[signal_key]
    previous_edged = stroke.props.get('edges', list())

    diff = np.diff(signal)
    indices = np.flatnonzero(np.abs(diff) > threshold)
    edge_types = np.where(diff[indices] > 0, 'rising', 'falling')
    current_edges = list(zip(edge_types, t[indices].tolist()))

    if current_edges[0][0] == 'falling':
        current_edges = current_edges[1:]

    if current_edges[-1][0] == 'rising':
        current_edges = current_edges[:-1]

    edges = list(heapq.merge(previous_edged, current_edges, key=lambda x: x[1]))
    return stroke.clone(props={'edges': edges}, force=True)
