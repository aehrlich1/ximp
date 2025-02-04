"""
This file will take care of all data related aspects.
"""
import torch
from torch.utils.data import Dataset


class AdmetPotencyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        smiles = self.dataset[idx]["smiles"]
        label = torch.tensor(self.dataset[idx]["label"], dtype=torch.float32)
        return smiles, label


class AdmetPotencyTestDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
