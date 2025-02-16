"""
This file will take care of all data related aspects.
"""
import csv
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_smiles
from torch.utils.data import Dataset

from src.utils import filter_and_extract, scaffold_split


class PotencyDataset(InMemoryDataset):
    def __init__(self, root, train=True, target_col=1, force_reload=False):
        self.target_col = target_col
        super().__init__(root, force_reload=force_reload)
        self.load(self.processed_paths[0] if train else self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ['train_polaris.csv', 'test_polaris.csv']

    @property
    def processed_file_names(self):
        return [f'train_{self.target_col}.pt', 'test.pt']

    def process(self):
        self.process_train() if self.train else self.process_test()

    def process_train(self):
        data_list: list[Data] = []
        with open(self.raw_paths[0], 'r') as file:
            lines = csv.reader(file)
            next(lines)  # skip header

            for line in lines:
                smiles = line[0]
                label = line[self.target_col]
                if len(label) == 0:
                    continue

                y = torch.tensor(float(label), dtype=torch.float).view(-1, 1)
                data = from_smiles(smiles)
                data.y = y
                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def process_test(self):
        data_list: list[Data] = []
        with open(self.raw_paths[1], 'r') as file:
            lines = csv.reader(file)
            next(lines)  # skip header

            for line in lines:
                smiles = line[0]
                data = from_smiles(smiles)
                data_list.append(data)

        self.save(data_list, self.processed_paths[1])


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


if __name__ == "__main__":
    dataset = PotencyDataset(root="../data/polaris/potency", train=True, target_col=1)
    train_dataset, test_dataset = scaffold_split(dataset)
    print(len(train_dataset), len(test_dataset))
    print(dataset)
