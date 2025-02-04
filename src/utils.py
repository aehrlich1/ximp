"""
This file will contain utility functions to be used by everybody
"""
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Batch


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
            print("Early stopping triggered.")

    def log(self, data: dict[str, int | float]) -> None:
        for key, value in data.items():
            attr = getattr(self, key)
            attr.append(value)
