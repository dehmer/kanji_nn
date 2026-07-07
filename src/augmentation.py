#!/usr/bin/env python3

import numpy as np
from kanji_nn.io import WKBReader
from kanji_nn.data import KanjiVGDataset, transform_absolute
from kanji_nn.plot import character

literal = lambda code_point: chr(int(code_point[2:], 16))
def filter_literals(literals): return lambda label: literal(label) in literals
downsample_candidates = "儷嚮囈囑囓嫋嬲孅屬嵶嶷巉巍巒巖廢廳彎慇懸懿戀搦撥擲擺攣攪斃斷曦朧櫞欅欒欝欟殲潺瀚瀛瀰灑灣燧燹爍爨犧獰獵獻癈癜癰癲發皺矚磐磯礙穢竅竈竊竸籐籔籘籠籤籬糶縫繦繼纃纉纎纏纒纓纔纖纛纜羆羸翳翹臘臚臟臠艤艨艫艷藝蘂蠡蠶蠻襲襷覊觸譌譎譏譖議變讒讓讖讚豫贓趨蹶躊躑躔軈轂轗轢轤遞遽邂邃邊邏鄒醫醵釀釁鏃鏐鏖鑁鑄鑓鑚鑞鑠鑢鑪鑰鑼鑽鑾鑿钁闥隧雛霧靄靆靉韃韆韈響顰顱飃飄飆飜餮饌饑饕饗馨驂驍驕驚驟驢驤驥驩驪驫髏髑髓髞鬟鬢鬣鬮鬱鬻魏魑魔魘鯱鰄鰯鰲鰺鱇鱒鱗鱧鱶鱸鵝鶚鶩鶯鶲鶸鷂鷄鷆鷙鷭鷯鷲鷸鷺鷽鸚鸛鸞麌麑麓麝麟黌黐黴黶黷鼇鼈齏齧齲齶齷龕龝"

def to_absolute(sequence, point_zero):
    raw = sequence.numpy()
    raw = raw[:, [0, 1, 3]]
    xy = point_zero + np.cumsum(raw[:,:2], axis=0)
    return np.hstack([xy, raw[:,2:]])

if __name__ == "__main__":
    DIR = "data/wkb"
    DATASET = "katakana_47"
    MAX_SEQUENCE_LENGTH = 256
    np.set_printoptions(precision=5, suppress=True)

    # Initialize dataset:
    reader = WKBReader(DIR, "kanken_6355", filter=filter_literals('巉'))
    dataset = KanjiVGDataset(reader, MAX_SEQUENCE_LENGTH, transform_absolute=transform_absolute)

    # time (no augmentation):             13.44s user 0.17s system 99% cpu 13.696 total
    # time (_character_to_tensor update): 12.71s user 0.15s system 99% cpu 12.932 total
    # Note: We are checking that no character exceeds point budget and
    # 255 expected candidates in full kanji kentei are downsampled.
    overshooting = []
    for sequence, label, point_zero in dataset:
        strokes = to_absolute(sequence, point_zero)
        character.show(strokes)
