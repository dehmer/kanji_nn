has_variant = lambda groups: any (x.startswith('VARIANT:') for x in groups)

hiragana_46 = lambda _, groups: groups == ['HIRAGANA']
hiragana_90 = lambda _, groups: 'HIRAGANA' in groups
katakana_47 = lambda _, groups: groups == ['KATAKANA']
katakana_90 = lambda _, groups: 'HIRAGANA' in groups

# 2,136 jōyō kanji excluding variants.
def kanji_joyo(literal, groups):
  if has_variant(groups): return False
  return all(x in groups for x in ["KANJI", "JOYO"])

# 6,355 kanken kanji excluding variants.
def kanji_kanken(literal, groups):
  if has_variant(groups): return False
  return all(x in groups for x in ["KANKEN"])
