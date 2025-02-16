from pathlib import Path

import numpy as np
import pandas as pd
import polaris as po
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import PotencyDataset
from src.models import PolarisModel, create_repr_model, create_proj_model
from src.utils import PerformanceTracker, scaffold_split


class PotencyDispatcher:
    """
    This class will have to iterate over the available target_cols as well
    """

    def __init__(self):
        pass


class Potency:
    def __init__(self, params: dict):
        self.params: dict = params
        self.performance_tracker = PerformanceTracker(Path("./models"), id_run="x")
        self.device: str = "cpu"
        self.competition = None
        self.train_polaris = None
        self.test_polaris = None
        self.train_scaffold = None
        self.test_scaffold = None
        self.loss_fn = None
        self.optimizer = None
        self.model = None

        self._init()

    def train(self, train_dataloader, valid_dataloader) -> None:
        for epoch in tqdm(range(self.params["epochs"])):
            self.performance_tracker.log({"epoch": epoch})
            self._train_loop(train_dataloader)
            self._valid_loop(valid_dataloader)

            self.performance_tracker.update_early_loss_state()
            if self.performance_tracker.early_stop:
                break

    def predict(self, model_weights_path: Path | None, dataloader) -> list[dict]:
        # 1. Take the model_weights_path and load the model
        # 2. Model weights correspond to a set of parameters
        # 3. Given a dataloader, make predictions on that dataloader
        # 4. If no model weights path is provided. Initialize a random model
        pass

    def _init(self):
        self._init_device()
        self._init_competition()
        self._init_dataset()
        self._init_model()
        self._init_optimizer()
        self._init_loss_fn()

    def _init_device(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def _init_competition(self):
        self.competition = po.load_competition("asap-discovery/antiviral-potency-2025")

    def _init_model(self):
        repr_model = create_repr_model(self.params)
        proj_model = create_proj_model(self.params)
        self.model = PolarisModel(repr_model, proj_model)

    def _init_loss_fn(self):
        self.loss_fn = nn.MSELoss()

    def _init_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.params['lr'])

    def _init_dataset(self):
        self.train_polaris = PotencyDataset(root='./data/polaris/potency', train=True, target_col=1)
        self.test_polaris = PotencyDataset(root='./data/polaris/potency', train=False)

        self.train_scaffold, self.test_scaffold = scaffold_split(dataset=self.train_polaris,
                                                                 test_size=self.params["scaffold_split_val_sz"])

    def run(self):
        smiles = self.train_scaffold.smiles
        labels = self.train_scaffold.y.view(-1).tolist()

        y_binned = pd.qcut(labels, q=self.params['num_cv_bins'], labels=False)
        skf = StratifiedKFold(n_splits=self.params['num_cv_folds'], shuffle=True, random_state=42)

        print('Running K-Fold CV...')
        val_loss_list = []

        for fold, (train_idx, valid_idx) in enumerate(skf.split(smiles, y_binned)):
            self._init_model()  # Reinitialize model
            self._init_optimizer()
            self.performance_tracker.reset()

            train_fold = self.train_scaffold[train_idx]
            valid_fold = self.train_scaffold[valid_idx]

            train_fold_dataloader = DataLoader(train_fold, batch_size=self.params['batch_size'], shuffle=True)
            valid_fold_dataloader = DataLoader(valid_fold, batch_size=self.params['batch_size'], shuffle=False)

            self.train(train_fold_dataloader, valid_fold_dataloader)
            val_loss_list.append(self.performance_tracker.best_valid_loss)

        print(f"Validation losses: {val_loss_list}")
        print(f"Average validation loss: {np.mean(val_loss_list)}")

        # Optimize the model with the best parameters
        # Return params and mean_val_loss for tracking performance for each hyperparams config

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
        epoch_loss = 0

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                out = self.model(data)
                loss = self.loss_fn(out, data.y)
                epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        self.performance_tracker.log({"valid_loss": average_loss})
