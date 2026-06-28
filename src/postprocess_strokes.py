#!/usr/bin/env python3

from os import walk
import numpy as np
import kanji_nn.plot as plot
import kanji_nn.plot_telemetry as plot_telemetry

def split_into_strokes(raw):
    # drop pen status and split into strokes
    split_indices = np.where(raw[:, 4] == 0)[0] + 1
    raw = raw[:, :-1]
    return np.split(raw, split_indices[:-1])


def compose_from_strokes(strokes):
    chunks = []
    for stroke in strokes:
        # pen status: 111....110
        status = np.ones((len(stroke), 1), dtype=np.float32)
        status[-1] = 0

        # insert status
        chunk = np.insert(stroke, [2], status, axis=1)
        chunks.append(chunk)
    return np.vstack(chunks)


def normalize_array(arr):
    """Normalizes a numpy array to the range [0-1], safely handling NaNs."""
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def calc_telemetry(strokes):
    TIME_STAMP, DX, DY, PRESSURE, FEATURE = 0, 1, 2, 3, 4
    diffs = np.diff(strokes, axis=0)
    dt, dx, dy = diffs[:, TIME_STAMP], diffs[:, DX], diffs[:, DY]

    distances = np.sqrt(dx**2 + dy**2)
    vectors = np.column_stack((dx, dy))
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    magnitudes = np.where(magnitudes == 0, 1.0, magnitudes)
    unit_vectors = vectors / magnitudes
    dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    velocities = distances / dt
    angles = np.degrees(np.arccos(dot_products))

    # padding
    velocities = velocities * 1000
    velocities = np.pad(velocities, (1, 0), mode='constant', constant_values=np.nan)
    angles = np.pad(angles, (1, 1), mode='constant', constant_values=np.nan)
    return np.vstack((strokes[:, TIME_STAMP], strokes[:, PRESSURE], velocities, angles)).T


def process(filename):
    # [timestamp[0]:x[1]:y[2]:pressure[3]:feature[4]]
    raw = np.load(f'strokes/{filename}')
    telemetry = calc_telemetry(raw)
    plot_telemetry(telemetry)
    exit()


if __name__ == "__main__":
    for (dirpath, dirnames, filenames) in walk('strokes'):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if not filename == 'U+3075.npy': continue
            process(filename)
