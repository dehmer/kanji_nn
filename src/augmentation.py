#!/usr/bin/env python3

import uuid
import numpy as np
from functools import reduce, partial
import torch
from kanji_nn.io import WKBReader
from kanji_nn.data import KanjiVGDataset, transform_absolute
from kanji_nn.plot import character
from kanji_nn.conditioning import rdp_to_budget_flat

point_count = lambda strokes: sum([len(stroke) for stroke in strokes])
literal = lambda code_point: chr(int(code_point[2:], 16))
def filter_literals(literals): return lambda label: literal(label) in literals
downsample_candidates = "儷嚮囈囑囓嫋嬲孅屬嵶嶷巉巍巒巖廢廳彎慇懸懿戀搦撥擲擺攣攪斃斷曦朧櫞欅欒欝欟殲潺瀚瀛瀰灑灣燧燹爍爨犧獰獵獻癈癜癰癲發皺矚磐磯礙穢竅竈竊竸籐籔籘籠籤籬糶縫繦繼纃纉纎纏纒纓纔纖纛纜羆羸翳翹臘臚臟臠艤艨艫艷藝蘂蠡蠶蠻襲襷覊觸譌譎譏譖議變讒讓讖讚豫贓趨蹶躊躑躔軈轂轗轢轤遞遽邂邃邊邏鄒醫醵釀釁鏃鏐鏖鑁鑄鑓鑚鑞鑠鑢鑪鑰鑼鑽鑾鑿钁闥隧雛霧靄靆靉韃韆韈響顰顱飃飄飆飜餮饌饑饕饗馨驂驍驕驚驟驢驤驥驩驪驫髏髑髓髞鬟鬢鬣鬮鬱鬻魏魑魔魘鯱鰄鰯鰲鰺鱇鱒鱗鱧鱶鱸鵝鶚鶩鶯鶲鶸鷂鷄鷆鷙鷭鷯鷲鷸鷺鷽鸚鸛鸞麌麑麓麝麟黌黐黴黶黷鼇鼈齏齧齲齶齷龕龝"


def compose(*functions):
    """Composes functions right to left."""
    return lambda x: reduce(lambda acc, f: f(acc), reversed(functions), x)


def absolute_of(sequence, point_zero):
    """
    Convert stroke sequence from relative to absolute space.
    """
    raw = sequence.numpy()
    raw = raw[:, [0, 1, 3]]
    xy = point_zero + np.cumsum(raw[:,:2], axis=0)
    return np.hstack([xy, raw[:,2:]])


# shape strokes to tensor sequence
def shape_tensor_sequence(strokes):
    rows = []
    for stroke in strokes:

        # calculate s(t) - normalized cumulative arc length:
        length = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(length)])
        s /= s[-1]

        # clamp (optional) and make hstack-friendly:
        s = np.clip(s, 0, 1).reshape(-1, 1)

        # prepare pen status: (n - 1) x '1' || '0'
        pen = np.ones((len(stroke), 1), dtype=np.float32)
        pen[-1] = 0 # pen-up
        rows.append(np.hstack((stroke, s, pen)))

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


def downsample(strokes, max_budget=256):
    # Downsample only if necessary:
    target_total = max_budget - len(strokes)
    count_total = point_count(strokes)
    if count_total > target_total:
        strokes, _, count_total = rdp_to_budget_flat(strokes, target_total=target_total)
        assert count_total <= target_total

    return strokes


def prepare_label_encoding(reader):
    labels = [entry.label for entry in reader.meta] # label offsets from CSV
    uniq_labels = sorted(list(set(labels)))
    encoded = {label: i for i, label in enumerate(uniq_labels)}
    encode = lambda idx: encoded[labels[idx]]
    label = lambda idx: labels[idx]
    return encode, label


if __name__ == "__main__":
    DIR = "data/wkb"
    DATASET = "katakana_47"
    MAX_SEQUENCE_LENGTH = 256
    np.set_printoptions(precision=5, suppress=True)

    reader = WKBReader(DIR, "kanken_6355")
    sample_count = len(reader)
    encode_label, lookup_label = prepare_label_encoding(reader)

    # label transform:
    target_transform = compose(
        lambda idx: torch.tensor(idx, dtype=torch.long),
        lambda idx: encode_label(idx % sample_count)
    )

    # strokes transform:
    transform = compose(
        torch.from_numpy,
        shape_tensor_sequence,
        transform_absolute,
        partial(downsample, max_budget=MAX_SEQUENCE_LENGTH),
        lambda idx: reader[idx][1]
    )

    # Drop-in replacement for `transform` to plot
    # and/or save transformed character to filesystem.
    def transform_proxy(idx):
        # pre-fetch strokes to catch point zero:
        strokes = reader[idx][1]
        point_zero = strokes[0][0]
        label = lookup_label(idx)

        transformed = transform(idx)
        strokes = absolute_of(transformed, point_zero)
        filename = f'data/transform/kanken_6355/png/{label}-{uuid.uuid4()}'
        character.save(filename, strokes) # save plot
        # character.show(strokes) # plot strokes
        return transformed

    # Initialize dataset:
    current_transform = transform_proxy
    current_target_transform = target_transform
    dataset = KanjiVGDataset(
        sample_count,
        transform = current_transform,
        target_transform = current_target_transform
    )

    # time (no augmentation):             13.44s user 0.17s system 99% cpu 13.696 total
    # time (_character_to_tensor update): 12.71s user 0.15s system 99% cpu 12.932 total
    # time (transform_absolute)           13.19s user 0.16s system 99% cpu 13.416 total

    for sequence, label in dataset:
        pass # to be continued....
