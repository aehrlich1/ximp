"""
This file will contain utility functions to be used by everybody
"""
from pathlib import Path

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


def scaffold_split(dataset: InMemoryDataset, test_size=0.2) -> tuple[InMemoryDataset, InMemoryDataset]:
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

        self.counter = 0
        self.patience = 5
        self.best_valid_loss = float("inf")
        self.early_stop = False

    def reset(self):
        # Resets state, required for use in CV
        self.epoch = []
        self.train_loss = []
        self.valid_loss = []
        self.test_pred = {}

        self.counter = 0
        self.patience = 5
        self.best_valid_loss = float("inf")
        self.early_stop = False

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

    def update_early_loss_state(self) -> None:
        if self.valid_loss[-1] < self.best_valid_loss:
            self.best_valid_loss = self.valid_loss[-1]
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            # print("Early stopping triggered.")

    def log(self, data: dict[str, int | float]) -> None:
        for key, value in data.items():
            attr = getattr(self, key)
            attr.append(value)


def load_yaml_to_dict(config_filename: str) -> dict:
    path = Path(".") / "config" / config_filename
    with open(path, "r") as file:
        config: dict = yaml.safe_load(file)

    return config


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
