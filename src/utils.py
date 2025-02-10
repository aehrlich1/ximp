"""
This file will contain utility functions to be used by everybody
"""
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
import yaml

from torch_geometric.utils import from_smiles


### Function to compute scaffold
def compute_scaffold(smiles):
    """Convert SMILES to scaffold using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Handle invalid SMILES
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold

def ScaffoldSplit(dataset, test_size=0.2):
    data = pd.DataFrame({
        "smiles": [x[0] for x in dataset],
        "target": [x[1] for x in dataset]
    })

    # Compute scaffolds for each molecule
    data["scaffold"] = data["smiles"].apply(compute_scaffold)

    # Group by scaffold
    scaffold_groups = data.groupby("scaffold").apply(lambda x: x.index.tolist())

    # Sort scaffolds by frequency (larger groups first for balanced splitting)
    scaffold_groups = sorted(scaffold_groups, key=len, reverse=True)

    # Prepare train, validation, and test sets
    train_scaffolds, test_scaffolds = train_test_split(scaffold_groups, test_size=0.2, random_state=42)
    train_scaffolds = sum(train_scaffolds, [])  # Flatten list
    test_scaffolds = sum(test_scaffolds, [])  # Flatten list

    # Create final splits
    train_data = data.loc[train_scaffolds].reset_index(drop=True)
    test_data = data.loc[test_scaffolds].reset_index(drop=True)
    train_data_list, test_data_list = [], []

    train_target_list = list(train_data['target'].values)
    for idx, smiles in enumerate(list(train_data['smiles'].values)):
        train_data_list.append((smiles, train_target_list[idx]))

    test_target_list = list(test_data['target'].values)
    for idx, smiles in enumerate(list(test_data['smiles'].values)):
        test_data_list.append((smiles, test_target_list[idx]))

    return train_data_list, test_data_list


def wrapped_forward(self, data):
    data = [from_smiles(x[0]) for x in data]
    data = Batch.from_data_list(data).to(self.device)
    x, edge_index, batch = data.x, data.edge_index, data.batch
    out = self._original_forward(x=x, edge_index=edge_index, batch=batch)
    return out


def custom_collate(batch):
    graphs = [b[0] for b in batch]  # Extract graph data
    targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)  # Extract targets
    return Batch.from_data_list(graphs), targets  # Return batched graph and target tensor

def filter_and_extract(polaris_train, target_col):
    return [
        {"smiles": polaris_train[i][0],
         "label": polaris_train[i][1][target_col]
         }
        for i in range(len(polaris_train))
        if not np.isnan(polaris_train[i][1][target_col])
    ]



def convert_numbers(obj):
    """Recursively convert numbers (including scientific notation) while preserving other types."""
    if isinstance(obj, dict):
        return {k: convert_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numbers(i) for i in obj]  # Convert list elements recursively
    elif isinstance(obj, str):
        # Convert valid numbers from string to int or float
        try:
            if "." in obj or "e" in obj or "E" in obj:  # Check for float or scientific notation
                return float(obj)
            else:
                return int(obj)
        except ValueError:
            return obj  # Return as-is if conversion fails
    return obj


class PerformanceTracker:
    def __init__(self, tracking_dir: Path, id_run: str):
        self.tracking_dir: Path = tracking_dir
        self.id_run = id_run
        self.epoch = []
        self.train_loss = []
        self.valid_loss = []
        self.test_pred = {}

        self.counter = 0
        self.patience = 15
        self.best_valid_loss = float("inf")
        self.early_stop = False

    def reset(self):
        # Resets state, required for use in CV
        self.epoch = []
        self.train_loss = []
        self.valid_loss = []
        self.test_pred = {}

        self.counter = 0
        self.patience = 15
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
            #print("Early stopping triggered.")

    def log(self, data: dict[str, int | float]) -> None:
        for key, value in data.items():
            attr = getattr(self, key)
            attr.append(value)


def load_yaml_to_dict(config_filename: str) -> dict:
    path = Path(".") / "config" / config_filename
    with open(path, "r") as file:
        config: dict = yaml.safe_load(file)

    return config
