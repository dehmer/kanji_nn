import torch
from torch.utils.data import Dataset
import numpy as np
from kanji_nn.conditioning import rdp_to_budget_flat, join_strokes
from kanji_nn.plot import character

point_count = lambda strokes: sum([len(stroke) for stroke in strokes])
literal = lambda code_point: chr(int(code_point[2:], 16))
defaultTransformAbsolute = lambda strokes: strokes

class KanjiVGDataset(Dataset):
    # TODO: Supply reader parameters and instantiate reader in __enter__
    def __init__(
        self,
        wkb_reader,
        max_sequence=256,
        transform_absolute=defaultTransformAbsolute
    ):
        # FIXME: reader closes with process ;-)
        # TODO: Dataset.__enter__/__exit__
        self.wkb_reader = wkb_reader
        self.max_sequence = max_sequence
        self.transform_absolute = transform_absolute
        self.downsampled = [] #  TODO: remove after use

    def __len__(self):
        return len(self.wkb_reader)

    def __getitem__(self, idx):
        label, strokes = self.wkb_reader[idx]
        print(literal(label))
        character.show(join_strokes(strokes))

        # Downsample if necessary:
        target_total = self.max_sequence - len(strokes)
        count_total = point_count(strokes)
        if count_total > target_total:
            self.downsampled.append(literal(label))
            strokes, _, count_total = rdp_to_budget_flat(strokes, target_total=target_total)
            assert count_total <= target_total

        transformedStrokes = self.transform_absolute(strokes)
        tensor = self._character_to_tensor(transformedStrokes)  # (total_points, 4) float32
        assert tensor.shape[0] == count_total + len(strokes)

        # idx is inherently also label index:
        idx = idx % len(self.wkb_reader)
        return torch.from_numpy(tensor), torch.tensor(idx, dtype=torch.long), strokes[0][0]

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
