#!/usr/bin/env python3

from kanji_nn.plot import character
from kanji_nn.conditioning import join_strokes, rdp_to_budget_weighted, rdp_to_budget_flat
from kanji_nn.io import load_wkb

if __name__ == '__main__':
    BASE_NAME = 'kanken_6355'
    target_total = 64

    # Iterate through the generator
    for data in load_wkb(BASE_NAME):
        index = data['index']
        label = data['label']
        strokes = data['strokes']
        joined_strokes = join_strokes(strokes)

        # weighted_simplified, _, _ = rdp_to_budget_weighted(strokes, target_total=target_total)
        weighted_simplified, _, _ = rdp_to_budget_flat(strokes, target_total=target_total)
        simplified_strokes = join_strokes(weighted_simplified)

        print(f"index: {index}, label: {label}, point counts: {simplified_strokes.shape[0]} / {joined_strokes.shape[0]}")
        character.show(joined_strokes)
        character.show(simplified_strokes)
