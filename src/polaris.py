from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import Adam, Optimizer
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from src.data import MoleculeNetDataset, PolarisDataset
from src.models import PolarisModel, create_proj_model, create_repr_model
from src.utils import PerformanceTracker, scaffold_split


class Polaris:
    def __init__(self, params: dict):
        self.params: dict = params
        self.performance_tracker = PerformanceTracker(Path("./models"), id_run="x")
        self.device: str = "cpu"
        self.train_polaris: InMemoryDataset
        self.test_polaris: InMemoryDataset
        self.train_scaffold: InMemoryDataset
        self.test_scaffold: InMemoryDataset
        self.loss_fn: nn.L1Loss
        self.optimizer: Optimizer
        self.model: nn.Module

        self._init()

    def _init(self):
        # self._init_device()
        self._init_dataset()
        self._init_model()
        self._init_optimizer()
        self._init_loss_fn()

    def run(self):
        smiles = self.train_scaffold.smiles
        labels = self.train_scaffold.y.view(-1).tolist()

        y_binned = pd.qcut(labels, q=self.params["num_cv_bins"], labels=False)
        skf = StratifiedKFold(n_splits=self.params["num_cv_folds"], shuffle=True, random_state=42)
        # scaffold_kfold = ScaffoldKFold(n_splits=5, shuffle=True, random_state=42)

        val_loss_list = []

        for train_idx, valid_idx in skf.split(smiles, y_binned):
            self._init_model()  # Reinitialize model
            self._init_optimizer()
            self.performance_tracker.reset()

            train_fold = self.train_scaffold[train_idx]
            valid_fold = self.train_scaffold[valid_idx]

            train_fold_dataloader = DataLoader(
                train_fold, batch_size=self.params["batch_size"], shuffle=True
            )
            valid_fold_dataloader = DataLoader(
                valid_fold, batch_size=self.params["batch_size"], shuffle=False
            )

            self.train(train_fold_dataloader, valid_fold_dataloader)
            val_loss_list.append(self.performance_tracker.best_valid_loss)

        self.params.update({"mean_val_loss": np.mean(val_loss_list)})
        self.params.update({"patience": self.performance_tracker.patience})
        self.params.update(
            {"final_avg_epochs": round(np.mean(self.performance_tracker.early_stop_epoch))}
        )

        # Check why epochs are output twice
        print(f"epochs per fold: {self.performance_tracker.early_stop_epoch}")

        # Reset model and train on train scaffold.
        # Evaluate on test scaffold. Report MAE.
        self._init_model()
        self._init_optimizer()
        self.train_final(self.train_scaffold)
        preds = self.predict(self.test_scaffold)
        preds = [pred[1] for pred in preds]
        mae = mean_absolute_error(preds, self.test_scaffold.y)
        self.params.update({"mae_test_scaffold": mae})

        print(f"Validation losses: {val_loss_list}")
        print(f"Average validation loss: {np.mean(val_loss_list)}")
        print(f"Mean absolute error for {self.params['target_task']} on test_scaffold: {mae:.3f}")

    def train(self, train_dataloader, valid_dataloader) -> None:
        for epoch in range(self.params["epochs"]):
            self.performance_tracker.log({"epoch": epoch})
            self._train_loop(train_dataloader)
            self._valid_loop(valid_dataloader)

            self.performance_tracker.update_early_loss_state()
            if self.performance_tracker.early_stop:
                self.performance_tracker.log({"early_stop_epoch": epoch})
                break

    def train_final(self, train_dataset) -> None:
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )
        for _ in range(self.params["final_avg_epochs"]):
            self._train_loop(train_dataloader)

    def _init_device(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _init_model(self):
        torch.manual_seed(seed=42)
        repr_model = create_repr_model(self.params)
        proj_model = create_proj_model(self.params)
        self.model = PolarisModel(repr_model, proj_model)

    def _init_loss_fn(self):
        self.loss_fn = nn.L1Loss()

    def _init_optimizer(self):
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )

    def _init_polaris_dataset(self):
        root = Path("./data") / "polaris" / self.params["task"]

        log_transform = True if self.params["task"] == "admet" else False

        self.train_polaris = PolarisDataset(
            root=root,
            task=self.params["task"],
            target_task=self.params["target_task"],
            train=True,
            log_transform=log_transform,
            force_reload=True,
            use_erg=self.params["use_erg"],
            use_ft=self.params["use_ft"],
            ft_resolution=self.params["ft_resolution"],
        )

        self.test_polaris = PolarisDataset(
            root=root,
            task=self.params["task"],
            target_task=self.params["target_task"],
            train=False,
            log_transform=log_transform,
            force_reload=True,
            use_erg=self.params["use_erg"],
            use_ft=self.params["use_ft"],
            ft_resolution=self.params["ft_resolution"],
        )

        self.train_scaffold, self.test_scaffold = scaffold_split(
            dataset=self.train_polaris, test_size=self.params["scaffold_split_val_sz"]
        )

    def _init_molecule_net_dataset(self):
        root = Path("./data") / "molecule_net"
        molecule_net_dataset = MoleculeNetDataset(
            root=root,
            target_task=self.params["target_task"],
            force_reload=False,
            use_erg=self.params["use_erg"],
            use_ft=self.params["use_ft"],
            ft_resolution=self.params["ft_resolution"],
        ).create_dataset()

        self.train_scaffold, self.test_scaffold = scaffold_split(
            dataset=molecule_net_dataset, test_size=self.params["scaffold_split_val_sz"]
        )

    def _init_dataset(self):
        match self.params["task"]:
            case "admet":
                self._init_polaris_dataset()
            case "potency":
                self._init_polaris_dataset()
            case "molecule_net":
                self._init_molecule_net_dataset()
            case _:
                raise NotImplementedError

    def _train_loop(self, dataloader):
        self.model.train()
        epoch_loss = 0

        for data in dataloader:
            data = data.to(self.device)
            out = self.model(data)
            loss = self.loss_fn(out, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        self.performance_tracker.log({"train_loss": average_loss})

    def _valid_loop(self, dataloader):
        self.model.eval()
        self.performance_tracker.valid_loss = []
        epoch_loss = 0

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                out = self.model(data)
                loss = self.loss_fn(out, data.y)
                epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        self.performance_tracker.log({"valid_loss": average_loss})

    def predict(self, dataset) -> list[tuple]:
        """
        Return a list, where each element is a tuple with the first element being the
        smiles string, and the second being the predicted value.
        """
        self.model.eval()
        smiles = dataset.smiles

        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        with torch.no_grad():
            data = next(iter(dataloader)).to(self.device)
            pred = self.model(data)

        pred = [p.item() for p in pred]

        return list(zip(smiles, pred))
