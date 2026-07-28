import re
import unicodedata
from pathlib import Path
import csv

# jōyō kanji:            2,136
# jinmeiyō kanji:          863
# kanji kentei (kanken): 6,355

# read kanji sets from TSV resource:
_DATA_FILE = Path(__file__).with_name("kanji_sets.tsv")
with _DATA_FILE.open(encoding="utf-8", newline="") as fp:
    sets = dict(csv.reader(fp, delimiter="\t"))

def sorted_ord_list(s):
    list = [ord(c) for c in s]
    list.sort() # in-place
    return list


def binary_search(l, x):
    if len(l) < 2:
        if l[0] == x: return True
        else:         return False
    mid = len(l) // 2
    if x < l[mid]: return binary_search(l[:mid], x)
    else:          return binary_search(l[mid:], x)


lookups = { k: sorted_ord_list(v) for k, v in sets.items() }

# small kana variants (捨て仮名 - sutegata)
sutegata = [
        # hiragana
        '\u3041\u3043\u3045\u3047\u3049', # ぁ, ぃ, ぅ, ぇ, ぉ
        '\u3063',                         # っ
        '\u3083\u3085\u3087',             # ゃ, ゅ, ょ
        '\u308E',                         # ゎ
        '\u3095\u3096',                   # ゕ, ゖ
        # katakana
        '\u30A1\u30A3\u30A5\u30A7\u30A9', # ァ, ィ, ゥ, ェ, ォ
        '\u30C3',                         # ッ
        '\u30E3\u30E5\u30E7',             # ャ, ュ, ョ
        '\u30EE',                         # ヮ
        '\u30F5\u30F6'                    # ヵ, ヶ
    ]

# TODO: SYMBOL
non_kanji = {
    'HIRAGANA':   lambda c: bool(re.match(r'^[\u3040-\u309F]+$', c)),
    'KATAKANA':   lambda c: bool(re.match(r'^[\u30A0-\u30FF]+$', c)),
    'DAKUTEN':    lambda c: '\u3099' in unicodedata.normalize('NFD', c),         # kana with 濁点 (dakuten - ゛)
    'HANDAKUTEN': lambda c: '\u309a' in unicodedata.normalize('NFD', c),         # kana with 半濁点 (handakuten - ゜)
    'SUTEGATA':   lambda c: bool(re.match(rf'^[{''.join(sutegata)}]+$', c)),
    'OBSOLETE':   lambda c: bool(re.match(r'^[\u3090\u3091\u30F0\u30F1]+$', c)), # ゐ, ゑ, ヰ and ヱ
    'KANJI':      lambda c: bool(re.match(r'^[\u4E00-\u9FFF]+$', c)),
    'DIGIT':      lambda c: bool(re.match(r'^[0-9\uFF10-\uFF19]+$', c)),
    'ROMAJI':     lambda c: bool(re.match(r'^[a-zA-Z\uFF21-\uFF3A]+$', c)),

    # hiragana: ゛ (dakuten), ゜ (handakuten), ゝ (kurikaeshi)
    # katakana: ・ (nakaguro), ヽ (katakana gaeshi)
    'OTHER':      lambda c: bool(re.match(r'^[\u309B\u309C\u309D\u30FB\u30FD\u30FE]+$', c))
}

def groups_from_ord(n):
    result = []
    for c, l in lookups.items():
        if binary_search(l, n):
            result.append(c)

    # add general 'KANKEN' if 'KANKEN_x'
    for c in result:
        if c.startswith('KANKEN_'):
            result.append('KANKEN')
            break

    return result

def groups_from_chr(literal):
    result = []
    for c, p in non_kanji.items():
        if p(literal):
            result.append(c)
    return result

def groups(literal: String):
    return groups_from_chr(literal) + groups_from_ord(ord(literal))
