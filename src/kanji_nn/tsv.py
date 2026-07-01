import ast

import numpy as np
from . import vg

def list_to_array_list_3d(list):
    return [np.asarray(item) for item in list]

def normalize_array_list(array_list):
    return [xs / 110 for xs in array_list]

def vstack_array_list_with_feature(array_list):
    features = []
    for stroke in array_list:
        feature = np.hstack((stroke, np.ones((len(stroke), 1))))
        feature[-1, 2] = 0  # Pen-Up Signal
        features.append(feature)

    return np.vstack(features)

def read(filename):
    literals = []
    labels = []
    all_strokes = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 4:
                continue

            literals.append(parts[0])
            labels.append(parts[1])

            list_3d = ast.literal_eval(parts[3])
            array_list = list_to_array_list_3d(list_3d)
            features = vstack_array_list_with_feature(array_list)
            all_strokes.append(features)

    return {
        'literals': literals,
        'labels': labels,
        'strokes': all_strokes
    }
