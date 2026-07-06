import numpy as np

def join_strokes(strokes):
    chunks = []
    for stroke in strokes:
        # pen status: 111....110
        status = np.ones((len(stroke), 1), dtype=np.float32)
        status[-1] = 0
        chunk = np.hstack((stroke, status))
        chunks.append(chunk)
    return np.vstack(chunks)
