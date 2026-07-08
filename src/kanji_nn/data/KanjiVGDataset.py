import torch
from torch.utils.data import Dataset


class KanjiVGDataset(Dataset):
    def __init__(self, sample_count, transform, target_transform, k=1):
        self.sample_count = sample_count
        self.transform = transform
        self.target_transform = target_transform
        self.k = k

    def __len__(self):
        return self.k * self.sample_count

    def __getitem__(self, idx):
        return self.transform(idx), self.target_transform(idx)
