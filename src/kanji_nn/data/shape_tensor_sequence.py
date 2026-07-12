import numpy as np

def shape_tensor_sequence(strokes):
    """
    Performs the following:
        1. s(t) calculation and h-stacking
        2. pen status preparation and h-stacking
        3. v-stacking all strokes to single matrix
        4. EOS handling (duplication -> 1/0)
        5. and delta x, y calculation and replacement
    """

    rows = []
    for stroke in strokes:

        # calculate s(t) - normalized cumulative arc length:
        length = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(length)])
        total = s[-1]
        s = s / total if total > 0 else np.zeros_like(s)

        # clamp (optional) and make hstack-friendly:
        s = np.clip(s, 0, 1).reshape(-1, 1)

        # prepare pen status: (n - 1) x '1' || '0'
        pen = np.ones((len(stroke), 1), dtype=np.float32)
        pen[-1] = 0 # pen-up
        rows.append(np.column_stack((stroke, s, pen)))

    # stich stroke points together:
    strokes = np.vstack(rows, dtype=np.float32)

    # duplicate EOS (end-of-stroke)
    PEN = 3 # pen-down/pen-up column after adding s(t):
    idx = np.where(strokes[:, PEN] == 0)[0]
    counts = np.ones(len(strokes), dtype=int)
    counts[idx] += 1
    strokes = np.repeat(strokes, counts, axis=0)

    # pen-down for previous EOS:
    idx = idx + np.arange(len(idx)) # account for shifts
    strokes[idx, PEN] = 1

    # calculate and replace Δx, Δy:
    deltas = np.diff(strokes[:, :2], axis=0)
    strokes[1:, :2] = deltas
    strokes[0, :2] = 0 # first row: Δx = Δy = 0 by definition
    return strokes
