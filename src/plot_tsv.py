#!/usr/bin/env python3

import os
import kanji_nn.tsv as tsv
import kanji_nn.plot as plot

MAX_POINTS = 80
SET_NAME = "hiragana_48"
TSV_FILE = f"/Users/dehmer/Public/Data/kanji_nn/tsv/{SET_NAME}.tsv"
test_set = tsv.read(TSV_FILE, MAX_POINTS)

if __name__ == '__main__':
    for i, literal in enumerate(test_set['literals']):
        label = test_set['labels'][i]
        strokes = test_set['strokes'][i]

        output_dir = f'images/{SET_NAME}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # plot.show(strokes, (5, 5))
        plot.save(f'{output_dir}/{label}', strokes, (5, 5))
