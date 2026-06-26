#!/usr/bin/env python3

import kanji_nn.tsv as tsv
import kanji_nn.plot as plot

MAX_POINTS = 80
TSV_FILE = "/Users/dehmer/Public/Data/kanken_439.tsv"
test_set = tsv.read(TSV_FILE, MAX_POINTS)

for i, literal in enumerate(test_set['literals']):
  if literal != '太': continue
  label = test_set['labels'][i]
  strokes = test_set['strokes'][i]

  plot.save(f'images/{label}', strokes)
  # plot.save(f'images/{label}-clean', cleanStrokes)
