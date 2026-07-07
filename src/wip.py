#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import csv
import shapely.wkb
from shapely.geometry import LineString, MultiLineString
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from kanji_nn.conditioning import rdp_to_budget_flat

literal = lambda code_point: chr(int(code_point[2:], 16))
filter_any = lambda label: True
def filter_literals(literals): return lambda label: literal(label) in literals
point_count = lambda strokes: sum([len(stroke) for stroke in strokes])

@dataclass
class IndexEntry:
    label:  str # e.g. 'U+4E00'
    offset: int # [byte]
    length: int # [byte]


class WKBReader:
    def __init__(self, dir_name, dataset_name, filter=filter_any):
        self.index_path = Path(f"{dir_name}/{dataset_name}.csv")
        self.wkb_path   = Path(f"{dir_name}/{dataset_name}.wkb")
        self.wkb        = self.wkb = self.wkb_path.open("rb")

        # Load index to memory and close file immediately.
        with self.index_path.open(newline="") as f:
            self.index = [
                # Note: We are dropping reduncant index column.
                IndexEntry(row["label"], int(row["offset"]), int(row["length"]))
                for row in csv.DictReader(f)
                if filter(row["label"])
            ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.wkb:
            self.wkb.close()
            self.wkb = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        self.wkb.seek(entry.offset)
        buffer = self.wkb.read(entry.length)

        # Parse WKB (Big-Endian) fixed MultiLineString.
        # Shapely automatically detects the Big-Endian header marker from the buffer.
        mls = shapely.wkb.loads(buffer)
        strokes = [np.array(ls.coords) for ls in mls.geoms]

        return entry.label, strokes

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class KanjiVGDataset(Dataset):
    # TODO: Supply reader parameters and instantiate reader in __enter__
    def __init__(self, wkb_reader, max_sequence=256):
        # FIXME: reader closes with process ;-)
        # TODO: Dataset.__enter__/__exit__
        self.wkb_reader = wkb_reader
        self.max_sequence = max_sequence
        self.downsampled = [] #  TODO: remove after use

    def __len__(self):
        return len(self.wkb_reader)

    def __getitem__(self, idx):
        label, strokes = self.wkb_reader[idx]

        # Downsample if necessary:
        target_total = self.max_sequence - len(strokes)
        if point_count(strokes) > target_total:
            self.downsampled.append(literal(label))
            strokes, _, sum_points = rdp_to_budget_flat(strokes, target_total=target_total)
            assert sum_points <= target_total

        tensor = self._character_to_tensor(strokes)  # (total_points, 4) float32

        # idx is inherently also label index:
        idx = idx % len(self.wkb_reader)
        return torch.from_numpy(tensor), torch.tensor(idx, dtype=torch.long)

    @staticmethod
    def _character_to_tensor(strokes):
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


if __name__ == "__main__":
    DIR = "data/wkb"
    DATASET = "katakana_47"
    MAX_SEQUENCE_LENGTH = 256

    # Know you API:
    # TODO: remove after use
    with WKBReader(DIR, DATASET) as reader:
        reader # 47, excluding ヰ and ヱ
        label, strokes = reader[0] # U+30A2 (ア), 2 strokes, 50 points
        label, strokes = reader[1] # U+30A2 (イ), 2 strokes, 27 points
        point_count(strokes) # 27
        labels = [label for label, _ in reader]
        ('U+30A2' in labels) # ア: True
        ('U+5B66' in labels) # 学: False (Kanji)
        len(labels) # 47

    # Initialize dataset:
    reader = WKBReader(DIR, "kanken_6355")
    dataset = KanjiVGDataset(reader, MAX_SEQUENCE_LENGTH)

    # time (no augmentation):             13.44s user 0.17s system 99% cpu 13.696 total
    # time (_character_to_tensor update): 12.71s user 0.15s system 99% cpu 12.932 total
    # Note: We are checking that no character exceeds point budget and
    # 255 expected candidates in full kanji kentei are downsampled.
    overshooting = []
    for sequence, label in dataset:
        if sequence.shape[0] > MAX_SEQUENCE_LENGTH:
            overshooting.append(label)

    downsample_candidates = "儷嚮囈囑囓嫋嬲孅屬嵶嶷巉巍巒巖廢廳彎慇懸懿戀搦撥擲擺攣攪斃斷曦朧櫞欅欒欝欟殲潺瀚瀛瀰灑灣燧燹爍爨犧獰獵獻癈癜癰癲發皺矚磐磯礙穢竅竈竊竸籐籔籘籠籤籬糶縫繦繼纃纉纎纏纒纓纔纖纛纜羆羸翳翹臘臚臟臠艤艨艫艷藝蘂蠡蠶蠻襲襷覊觸譌譎譏譖議變讒讓讖讚豫贓趨蹶躊躑躔軈轂轗轢轤遞遽邂邃邊邏鄒醫醵釀釁鏃鏐鏖鑁鑄鑓鑚鑞鑠鑢鑪鑰鑼鑽鑾鑿钁闥隧雛霧靄靆靉韃韆韈響顰顱飃飄飆飜餮饌饑饕饗馨驂驍驕驚驟驢驤驥驩驪驫髏髑髓髞鬟鬢鬣鬮鬱鬻魏魑魔魘鯱鰄鰯鰲鰺鱇鱒鱗鱧鱶鱸鵝鶚鶩鶯鶲鶸鷂鷄鷆鷙鷭鷯鷲鷸鷺鷽鸚鸛鸞麌麑麓麝麟黌黐黴黶黷鼇鼈齏齧齲齶齷龕龝"
    assert len(downsample_candidates) == len(dataset.downsampled)
    assert len(overshooting) == 0
