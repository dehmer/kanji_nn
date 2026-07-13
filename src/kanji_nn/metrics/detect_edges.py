import heapq
import numpy as np
from collections import OrderedDict
from itertools import groupby


def drop_same_adjacent(edges):
    result = []

    for key, group in groupby(edges, key=lambda x: x[0]):
        group_list = list(group)

        if key == "falling":
            result.append(group_list[0])  # Keep first
        elif key == "rising":
            result.append(group_list[-1]) # Keep last
        else:
            result.extend(group_list)     # Keep all if other value

    return result


def filter_contradictions(edges):
    # 1. Gruppieren nach der zweiten Komponente (Reihenfolge bleibt erhalten)
    groups = OrderedDict()
    for first, second in edges:
        if second not in groups:
            groups[second] = []
        groups[second].append((first, second))

    result = []

    # 2. Gruppen nach Ihren Regeln filtern
    for second, group in groups.items():
        if len(group) == 1:
            # Regel 3: Einzelne Tupel direkt hinzufügen
            result.append(group[0])
        else:
            # Überprüfen, ob alle ersten Komponenten in der Gruppe identisch sind
            first_components = {t[0] for t in group}

            if len(first_components) == 1:
                # Regel 2: Wenn identisch, nur eines der Tupel hinzufügen
                result.append(group[0])
            # Regel 1: Wenn nicht identisch, wird die gesamte Gruppe verworfen

    return result


def detect_edges(stroke, signal_key, threshold=0.9):
    signal = stroke.features[signal_key]
    previous_edged = stroke.props.get('edges', list())

    diff = np.diff(signal)
    indices = np.flatnonzero(np.abs(diff) > threshold)
    edge_types = np.where(diff[indices] > 0, 'rising', 'falling')
    current_edges = list(zip(edge_types, indices.tolist()))


    edges = list(heapq.merge(previous_edged, current_edges, key=lambda x: x[1]))
    edges = drop_same_adjacent(edges)
    edges = filter_contradictions(edges)

    # drop leading edge if falling:
    if len(edges) and edges[0][0] == 'falling':
        edges = edges[1:]

    # drop trailing edge if rising:
    if len(edges) and edges[-1][0] == 'rising':
        edges = edges[:-1]

    return stroke.clone(props={'edges': edges}, force=True)
