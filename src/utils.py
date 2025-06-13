"""
This file will contain utility functions to be used by everybody
"""
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


class ScaffoldKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, smiles_list, y=None):
        """Generate Scaffold-based KFold splits."""
        # Step 1: Compute scaffolds
        scaffold_to_indices = defaultdict(list)
        for idx, smiles in enumerate(smiles_list):
            scaffold = generate_scaffold(smiles)
            scaffold_to_indices[scaffold].append(idx)

        # Step 2: Shuffle scaffolds (if required)
        scaffold_groups = list(scaffold_to_indices.values())
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(scaffold_groups)

        # Step 3: Split scaffolds into K folds
        fold_indices = [[] for _ in range(self.n_splits)]
        for i, group in enumerate(scaffold_groups):
            fold_indices[i % self.n_splits].extend(group)

        # Step 4: Yield train/validation indices
        for i in range(self.n_splits):
            train_idx = [idx for j, fold in enumerate(fold_indices) if j != i for idx in fold]
            valid_idx = fold_indices[i]
            yield np.array(train_idx), np.array(valid_idx)


def generate_scaffold(smiles) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"{smiles} is not a valid SMILES. Could not generate scaffold. Returning None.")
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold


def scaffold_split(
    dataset: InMemoryDataset, test_size=0.2
) -> tuple[InMemoryDataset, InMemoryDataset]:
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

        self.best_valid_loss = float("inf")

    def reset(self):
        self.epoch = []
        self.train_loss = []
        self.valid_loss = []
        self.test_pred = {}

        self.best_valid_loss = float("inf")

    def save_performance(self) -> None:
        self.save_to_csv()
        self.save_loss_plot()

    def save_loss_plot(self) -> None:
        loss_plot_path = self.tracking_dir / f"{self.id_run}_loss.pdf"
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained", dpi=300)
        ax.plot(self.epoch, self.train_loss, self.valid_loss)
        ax.grid(True)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Valid"])

        fig.savefig(loss_plot_path)
        print(f"Saved loss plot to: {loss_plot_path}")

    def save_to_csv(self) -> None:
        df = pd.DataFrame(
            {
                "epoch": self.epoch,
                "train_loss": self.train_loss,
                "valid_loss": self.valid_loss,
            }
        )
        df.to_csv(self.tracking_dir / f"{self.id_run}.csv", index=False)

        results = pd.DataFrame(self.test_pred)
        results.to_csv(self.tracking_dir / f"{self.id_run}_results.csv", index=False)

    def get_results(self) -> dict[str, float]:
        return {
            "train_loss": self.train_loss[-1],
            "valid_loss": self.valid_loss[-1],
        }

    def update_best_valid_loss(self) -> None: # Deprecated/unused
        if self.valid_loss[-1] < self.best_valid_loss:
            self.best_valid_loss = self.valid_loss[-1]

    def log(self, data: dict[str, int | float]) -> None:
        for key, value in data.items():
            attr = getattr(self, key)
            attr.append(value)


def load_yaml_to_dict(config_filename: str) -> dict:
    path = Path(".") / "config" / config_filename
    with open(path, "r") as file:
        config: dict = yaml.safe_load(file)

    return config


def make_combinations_improved(
    dictionary: dict, coupled_keys: tuple[str, str] = None
) -> list[dict]:
    # Start with an empty combination
    combinations = [{}]

    # Handle regular (non-coupled) keys
    for key, value in dictionary.items():
        # Skip coupled keys as they're handled separately
        if coupled_keys and key in coupled_keys:
            continue

        # Convert single values to lists for consistent handling
        values = [value] if not isinstance(value, list) else value

        # Create new combinations with each value
        combinations = [
            {**existing, key: new_value} for existing in combinations for new_value in values
        ]

    # Handle coupled keys if provided
    if coupled_keys and all(k in dictionary for k in coupled_keys):
        key1, key2 = coupled_keys
        # Get values for both keys
        vals1 = dictionary[key1] if isinstance(dictionary[key1], list) else [dictionary[key1]]
        vals2 = dictionary[key2] if isinstance(dictionary[key2], list) else [dictionary[key2]]
        # Pair the values
        paired_values = list(zip(vals1, vals2))

        # Add coupled values to all combinations
        combinations = [
            {**existing, key1: val1, key2: val2}
            for existing in combinations
            for val1, val2 in paired_values
        ]

    return combinations


def make_combinations(dictionary: dict, exclude_key: str = None) -> list[dict]:
    # Start with first key-value pair
    result = [{}]

    for key, value in dictionary.items():
        if key == exclude_key:
            # Add this key with its list value to all existing dictionaries
            for r in result:
                r[key] = value
        else:
            # For all other keys, create combinations
            new_result = []
            values = [value] if not isinstance(value, list) else value
            for r in result:
                for v in values:
                    new_dict = r.copy()
                    new_dict[key] = v
                    new_result.append(new_dict)
            result = new_result

    return result


def save_dict_to_csv(data: list[dict], output_path: Path):
    with open(output_path, "w", newline="") as file:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def save_dict_to_yaml(data: dict, output_path: Path):
    with open(output_path, "w") as f:
        yaml.dump(data, f)


def format_time_readable(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60
    return f"{hours}h {minutes}m {sec}s"

def str2bool(v):
    # Convert a string representation of truth to a boolean value.
    # Necessary for argparse to handle boolean flags correctly.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
