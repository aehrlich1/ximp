import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import InMemoryDataset


def generate_scaffold(smiles) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"{smiles} is not a valid SMILES. Could not generate scaffold. Returning None.")
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold


def scaffold_split(dataset: InMemoryDataset, test_size=0.2):
    """
    Apply a mask to the provided dataset according to their scaffold groups.
    Return a train/test scaffold split.
    """

    # Group molecule indices by their scaffolds
    scaffold_groups = {}
    for idx, data in enumerate(dataset):
        scaffold = generate_scaffold(data.smiles)
        scaffold_groups.setdefault(scaffold, []).append(idx)

    # Sort groups by size, largest first
    sorted_groups = sorted(scaffold_groups.values(), key=len, reverse=True)

    # Split into train/test while keeping scaffolds together
    train_size = int(len(dataset) * (1 - test_size))
    train_idx = []
    test_idx = []

    for group in sorted_groups:
        if len(train_idx) + len(group) <= train_size:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    return dataset[train_idx], dataset[test_idx]


class PerformanceTracker:
    def __init__(self, tracking_dir: Path, id_run: str):
        self.tracking_dir: Path = tracking_dir
        self.id_run = id_run
        self.epoch = []
        self.train_loss = []
        self.valid_loss = []
        self.test_pred = {}

    def reset(self):
        self.epoch = []
        self.train_loss = []
        self.valid_loss = []
        self.test_pred = {}

    def log(self, data: dict[str, int | float]) -> None:
        for key, value in data.items():
            attr = getattr(self, key)
            attr.append(value)


def save_dict_to_csv(data: list[dict], output_path: Path):
    with open(output_path, "w", newline="") as file:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
