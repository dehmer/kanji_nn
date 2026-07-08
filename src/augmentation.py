#!/usr/bin/env python3

import numpy as np
from functools import partial
import torch
from kanji_nn.io import WKBReader
from kanji_nn.data import KanjiVGDataset, transform_absolute
from kanji_nn.data import s_weighted_random_walk_noise, shape_tensor_sequence
from kanji_nn.data import prepare_label_encoding, compose
from kanji_nn.plot import character
from kanji_nn.conditioning import rdp_to_budget_flat, absolute_of, downsample

literal = lambda code_point: chr(int(code_point[2:], 16))
def filter_literals(literals): return lambda label: literal(label) in literals
downsample_candidates = "儷嚮囈囑囓嫋嬲孅屬嵶嶷巉巍巒巖廢廳彎慇懸懿戀搦撥擲擺攣攪斃斷曦朧櫞欅欒欝欟殲潺瀚瀛瀰灑灣燧燹爍爨犧獰獵獻癈癜癰癲發皺矚磐磯礙穢竅竈竊竸籐籔籘籠籤籬糶縫繦繼纃纉纎纏纒纓纔纖纛纜羆羸翳翹臘臚臟臠艤艨艫艷藝蘂蠡蠶蠻襲襷覊觸譌譎譏譖議變讒讓讖讚豫贓趨蹶躊躑躔軈轂轗轢轤遞遽邂邃邊邏鄒醫醵釀釁鏃鏐鏖鑁鑄鑓鑚鑞鑠鑢鑪鑰鑼鑽鑾鑿钁闥隧雛霧靄靆靉韃韆韈響顰顱飃飄飆飜餮饌饑饕饗馨驂驍驕驚驟驢驤驥驩驪驫髏髑髓髞鬟鬢鬣鬮鬱鬻魏魑魔魘鯱鰄鰯鰲鰺鱇鱒鱗鱧鱶鱸鵝鶚鶩鶯鶲鶸鷂鷄鷆鷙鷭鷯鷲鷸鷺鷽鸚鸛鸞麌麑麓麝麟黌黐黴黶黷鼇鼈齏齧齲齶齷龕龝"


if __name__ == "__main__":
    DIR = "data/wkb"
    # DATASET = "katakana_47"
    DATASET = "kanken_6355"
    MAX_SEQUENCE_LENGTH = 256
    MULTIPLIER = 1
    np.set_printoptions(precision=5, suppress=True)

    filter = filter_literals("オ")
    reader = WKBReader(DIR, DATASET)
    # reader = WKBReader(DIR, DATASET, filter=filter)
    sample_count = len(reader)
    encode_label, lookup_label = prepare_label_encoding(reader)

    # label transform:
    target_transform = compose(
        lambda idx: torch.tensor(idx, dtype=torch.long),
        lambda idx: encode_label(idx % sample_count)
    )

    # strokes transform:
    def transform(idx):
        """
        k:   [0.6, 1.0]
        rho: [0, 0.5]
        """
        # k = np.linspace(0.6, 1.0, num=MULTIPLIER)[idx % MULTIPLIER]
        # rho = np.linspace(0, 0.7, num=MULTIPLIER)[idx % MULTIPLIER]

        random_walk_noise = partial(s_weighted_random_walk_noise, sigma_base=0.03, k=1, rho=0.5)
        return compose(
            torch.from_numpy,
            random_walk_noise,
            shape_tensor_sequence,
            transform_absolute,
            partial(downsample, max_budget=MAX_SEQUENCE_LENGTH),
            lambda idx: reader[idx % sample_count][1]
        )(idx)

    # Drop-in replacement for `transform` to plot
    # and/or save transformed character to filesystem.
    def transform_proxy(idx):
        # pre-fetch strokes to catch point zero:
        strokes = reader[idx % sample_count][1]
        point_zero = strokes[0][0]
        label = lookup_label(idx % sample_count)
        # print(literal(label))

        transformed = transform(idx)
        strokes = absolute_of(transformed, point_zero)
        filename = f'data/transform/kanken_6355/png/{label}-{idx % sample_count},{idx % MULTIPLIER}'
        # character.save(filename, strokes) # save plot
        # character.show(strokes) # plot strokes
        return transformed

    # Initialize dataset:
    current_transform = transform_proxy
    current_target_transform = target_transform
    dataset = KanjiVGDataset(
        sample_count,
        transform = current_transform,
        target_transform = current_target_transform,
        k=MULTIPLIER
    )

    # time (no augmentation)        13.44s user 0.17s system 99% cpu 13.696 total
    # time (shape_tensor_sequence)  12.71s user 0.15s system 99% cpu 12.932 total
    # time (transform_absolute)     13.19s user 0.16s system 99% cpu 13.416 total
    # time (random walk noise)      15.30s user 0.18s system 98% cpu 15.642 total

    for idx in range(len(dataset)):
        sequence, label = dataset[idx]
        # print(idx, label)
        pass # to be continued....
