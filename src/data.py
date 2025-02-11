"""
This file will take care of all data related aspects.
"""
import torch
from torch.utils.data import Dataset

from src.utils import filter_and_extract


class AntiviralPotencyDataset(Dataset):
    """
    Only takes a single prediction value.
    """

    def __init__(self, dataset):
        # Remove all entries that are NaN
        filtered_dataset = filter_and_extract(dataset, "pIC50 (MERS-CoV Mpro)")
        self.dataset = filtered_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        smiles = self.dataset[idx]["smiles"]
        label = torch.tensor(self.dataset[idx]["label"], dtype=torch.float32)
        return smiles, label


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
