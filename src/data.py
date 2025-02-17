"""
This file will take care of all data related aspects.
"""
import csv
import warnings

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_smiles

from src.utils import scaffold_split

# Ignore FutureWarnings from torch.load about weightsOnly bool != True
warnings.simplefilter("ignore", category=FutureWarning)


class PotencyDataset(InMemoryDataset):
    def __init__(self, root, train=True, target_task: str="pIC50 (MERS-CoV Mpro)", force_reload=False):
        self.target_task = target_task
        self.target_col = self._admet_target_to_col_mapping(target_task)
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

    @staticmethod
    def _admet_target_to_col_mapping(target_task: str) -> int:
        if target_task == "pIC50 (MERS-CoV Mpro)":
            return 1
        elif target_task == "pIC50 (SARS-CoV-2 Mpro)":
            return 2
        else:
            raise ValueError(f"Unknown target task: {target_task}")


if __name__ == "__main__":
    dataset = PotencyDataset(root="../data/polaris/potency", train=True)
    train_dataset, test_dataset = scaffold_split(dataset)
    print(len(train_dataset), len(test_dataset))
    print(dataset)
