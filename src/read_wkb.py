#!/usr/bin/env python3

import numpy as np
from kanji_nn import parse_archive
from kanji_nn.filters import hiragana_90, hiragana_46, kanji_kanken
from kanji_nn.plot import character
from kanji_nn.conditioning import *
from kanji_nn.io import load_wkb

def join_strokes(strokes):
    chunks = []
    for stroke in strokes:
        # pen status: 111....110
        status = np.ones((len(stroke), 1), dtype=np.float32)
        status[-1] = 0
        chunk = np.hstack((stroke, status))
        chunks.append(chunk)
    return np.vstack(chunks)


if __name__ == '__main__':
    base_name = 'kanken_6355'

    # Iterate through the generator
    for literal in load_wkb(base_name):
        index = literal['index']
        label = literal['label']
        strokes = literal['strokes']
        strokes = [simplify_rdp(stroke) for stroke in strokes]
        strokes = join_strokes(strokes)
        point_count = strokes.shape[0]
        # character.show(strokes)
        print(f"index: {index}, label: {label}, point count: {point_count}")
