import torch
from torch.utils.data import Dataset
import numpy as np

from . import parse_archive

def split_blocks(rows):
    ends = np.where(rows[:, 2] == 0)[0]
    return np.split(rows, ends[:-1] + 1)

def transform_block(block):
    coords = block[:, :2]
    z_col = block[:, 2:]

    translate = 0.02
    alpha = 7
    scale = 0.3
    cx, cy = np.mean(coords, axis=0)
    tx = np.random.uniform(-translate, translate)
    ty = np.random.uniform(-translate, translate)
    angle_rad = np.radians(np.random.uniform(-alpha, alpha))
    sx = np.random.uniform(1 - scale, 1 + scale)
    sy = np.random.uniform(1 - scale, 1 + scale)
    cs, sn = np.cos(angle_rad), np.sin(angle_rad)

    TC = np.array([[1,    0,     -cx], [0,    1,     -cy], [0, 0, 1]])
    S  = np.array([[sx,   0,       0], [0,   sy,       0], [0, 0, 1]])
    R  = np.array([[cs, -sn,       0], [sn,  cs,       0], [0, 0, 1]])
    TO = np.array([[1,    0, cx + tx], [0,    1, cy + ty], [0, 0, 1]])
    M = TO @ R @ S @ TC

    # Convert 2D coordinates to Homogeneous coordinates (N x 3)
    num_points = coords.shape[0]
    ones_col = np.ones((num_points, 1))
    hom_coords = np.hstack((coords, ones_col))
    trans_hom_coords = hom_coords @ M.T
    trans_coords = trans_hom_coords[:, :2]
    return np.hstack((trans_coords, z_col))


class KanjiVGDataset(Dataset):
    def __init__(
      self,
      filename,
      filter,
      n_points,
      multiplier = 32
    ):
        all_strokes, labels = parse_archive(filename, filter, n_points)
        self.all_strokes = all_strokes
        self.labels = labels
        self.multiplier = multiplier

        # label mapping: label -> index and index -> label
        self.label_indices = { label: idx for idx, label in enumerate(labels) }
        self.encoded_labels = [self.label_indices[label] for label in labels]

    def __len__(self):
        return len(self.labels) * self.multiplier

    def __getitem__(self, idx):
        base_idx = idx % len(self.labels)
        rows = self.all_strokes[base_idx].copy()
        blocks = split_blocks(rows)
        trans_blocks = [transform_block(block) for block in blocks]
        trans_rows = np.vstack(trans_blocks)
        encoded_label = self.encoded_labels[base_idx]
        return torch.tensor(trans_rows, dtype=torch.float32), torch.tensor(encoded_label, dtype=torch.long)
