from torch.utils.data import DataLoader, Subset

from src.data import AntiviralPotencyDataset
from src.models import PolarisModel, create_repr_model, create_proj_model
from src.utils import PerformanceTracker, ScaffoldSplit
import polaris as po
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


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
        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
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
        competition_data_dir: Path = Path(f"./data/antiviral-potency-2025")
        self.competition.cache(competition_data_dir)

    def _init_model(self):
        # Create the model based on the params
        repr_model = create_repr_model(self.params)
        proj_model = create_proj_model(self.params)
        self.model = PolarisModel(repr_model, proj_model)

    def _init_loss_fn(self):
        self.loss_fn = nn.MSELoss()

    def _init_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.params['lr'])

    def _init_dataset(self):
        train_dataset_polaris, test_dataset_polaris = self.competition.get_train_test_split()
        self.test_dataset_polaris = test_dataset_polaris

        # Filter values
        potency_dataset = AntiviralPotencyDataset(train_dataset_polaris)
        self.dataset = potency_dataset

        self.train_dataset, self.valid_dataset = ScaffoldSplit(dataset=self.dataset,
                                                               test_size=self.params["scaffold_split_val_sz"])

    def run(self):
        smiles = [X[0] for X in self.train_dataset]
        labels = [X[1] for X in self.train_dataset]

        y_binned = pd.qcut(labels, q=self.params['num_cv_bins'], labels=False)
        skf = StratifiedKFold(n_splits=self.params['num_cv_folds'], shuffle=True, random_state=42)

        print('Running K-Fold CV...')
        val_loss_list = []

        for fold, (train_idx, valid_idx) in enumerate(skf.split(smiles, y_binned)):
            self._init_model() # Reinitialize model
            self._init_optimizer()
            self.performance_tracker.reset()

            subset_train = Subset(self.train_dataset, train_idx)
            subset_valid = Subset(self.train_dataset, valid_idx)

            subset_train_dataloader = DataLoader(subset_train, batch_size=self.params['batch_size'], shuffle=True)
            subset_valid_dataloader = DataLoader(subset_valid, batch_size=self.params['batch_size'], shuffle=False)

            self.train(subset_train_dataloader, subset_valid_dataloader)
            val_loss_list.append(self.performance_tracker.best_valid_loss)

        mean_val_loss = np.mean(val_loss_list)
        print(f"Average validation loss: {mean_val_loss}")

        # TODO Optimize the model with the best parameters

    def _train_loop(self, dataloader):
        self.model.train()
        epoch_loss = 0
        for batch, (smiles, labels) in enumerate(dataloader):
            labels = labels.view(-1, 1).to(self.device)
            out = self.model(smiles)
            loss = self.loss_fn(out, labels)
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
            for smiles, labels in dataloader:
                labels = labels.view(-1, 1).to(self.device)
                out = self.model(smiles)
                loss = self.loss_fn(out, labels)
                epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        self.performance_tracker.log({"valid_loss": average_loss})
