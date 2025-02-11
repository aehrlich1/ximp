from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import polaris as po
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.utils import from_smiles
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.data import AdmetPotencyDataset, AdmetPotencyTestDataset
from src.models import AdmetPotencyModel, HIMPModel
from src.transform import OGBTransform, JunctionTree
from src.utils import filter_and_extract, PerformanceTracker, custom_collate, compute_scaffold, ScaffoldSplit, \
    convert_numbers
import yaml


class Dispatcher:
    def __init__(self, params: dict):
        self.params = params


class Potency:
    """
    Potency takes a dictionary of params for a single run. I.e this class
    is not responsible for iterating through hyperparameter values.

    Example usage:
    params = {
      lr: 0.01
      epochs: 20
    }

    Incorrect usage:
    params = {
      lr: [0.01, 0.001]
      epochs: 20
    }
    """

    def __init__(self, params: dict):
        self.params: dict = params
        self.performance_tracker = PerformanceTracker()
        self.competition = None
        self.dataset = None
        self.loss_fn = None
        self.optimizer = None
        self.model = None
        self.device = "cpu"

    def _init_competition(self):
        pass

    def _init_device(self):
        pass

    def _init_loss_fn(self):
        self.loss_fn = nn.MSELoss()

    def _init_model(self):
        pass

    def _init_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.params['lr'])


class AdmetPotency:
    def __init__(self, params: dict, performance_tracker: PerformanceTracker):
        self.params = params
        self.performance_tracker = performance_tracker
        self.competition = None
        self.dataset = None
        self.loss_fn = None
        self.optimizer = None
        self.model = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize()

    def _initialize(self):
        self._initialize_loss_fn()
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_data()

    def _initialize_loss_fn(self):
        self.loss_fn = nn.MSELoss()

    def _initialize_model(self, latent_dim=128, projector_dim=64, repr_himp_hidden_dim=9, rep_himp_num_layers=2,
                          rep_himp_dropout=0.5, rep_himp_inter=True, repr_type='HIMP'):
        self.model = AdmetPotencyModel(latent_dim=latent_dim, projector_dim=projector_dim,
                                       repr_himp_hidden_dim=repr_himp_hidden_dim,
                                       rep_himp_num_layers=rep_himp_num_layers, rep_himp_dropout=rep_himp_dropout,
                                       rep_himp_inter=rep_himp_inter,
                                       repr_type=repr_type)  # TODO Make kwargs (also in other methods with long kw lists)
        self.model.to(self.device)

    def _initialize_optimizer(self, lr=0.001):
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def _initialize_data(self):
        # TODO: Integrate in config or remove bc deprecated
        self.competition = po.load_competition(f"asap-discovery/{self.params["task"]}-2025")
        data_dir: Path = Path("../data/antiviral-{}-2025".format(self.params['task']))
        self.competition.cache(data_dir)

        train, test = self.competition.get_train_test_split()
        data = filter_and_extract(train, target_col=self.params["target_col"])
        dataset = AdmetPotencyDataset(data)
        test_dataset = AdmetPotencyTestDataset(test)

        train_dataset, valid_dataset = random_split(dataset=dataset, lengths=[0.9, 0.1])
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.params['batch_size'][0], shuffle=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.params['batch_size'][0], shuffle=False,

        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False,
        )

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

    def cross_validate(self):
        self.competition = po.load_competition(f"asap-discovery/{self.params["task"]}-2025")
        data_dir: Path = Path(f"./data/{self.params["task"]}-2025")
        self.competition.cache(data_dir)

        train, test = self.competition.get_train_test_split()
        data = filter_and_extract(train, target_col=self.params["target_col"])
        dataset = AdmetPotencyDataset(data)

        train_dataset, valid_dataset = ScaffoldSplit(dataset=dataset, test_size=self.params['scaffold_split_val_sz'])

        test_dataset = AdmetPotencyTestDataset(test)
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        self.test_dataloader = test_dataloader

        y, X = [x[1] for x in train_dataset], [x[0] for x in train_dataset]
        num_bins = self.params['num_cv_bins']  # Adjust based on dataset size
        y_binned = pd.qcut(y, q=num_bins, labels=False)
        skf = StratifiedKFold(n_splits=self.params['num_cv_folds'], shuffle=True, random_state=42)

        # Example: learning rate & batch size optimization via GridSearch
        if self.params['repr_type'] == "ECFP":
            combinations = list(product(self.params['lr'], self.params['batch_size'], self.params['latent_dim'],
                                        self.params['projector_dim'], [None], [None], [None], [None]))
        elif self.params['repr_type'] in ["HIMP", "GIN", "GCN"]:
            combinations = list(product(self.params['lr'], self.params['batch_size'], self.params['latent_dim'],
                                        self.params['projector_dim'],
                                        self.params['repr_himp_hidden_dim'], self.params['rep_himp_num_layers'],
                                        self.params['rep_himp_dropout'], self.params['rep_himp_inter']
                                        ))
        best_mean_val_loss = float('inf')
        best_combination = None
        print('Optimizing hyperparameters via k-fold CV')
        for combination_idx, combination in enumerate(combinations):
            print('Combination {} of {}'.format(combination_idx + 1, len(combinations)))
            val_loss_list = []
            for fold, (train_idx, valid_idx) in enumerate(
                    skf.split(X, y_binned)):  # tqdm(enumerate(skf.split(X, y_binned))):
                # Reinitialize stateful components
                self.performance_tracker.reset()
                self._initialize_model(combination[2], combination[3], combination[4], combination[5], combination[6],
                                       combination[7], repr_type=self.params['repr_type'])
                self._initialize_optimizer(lr=combination[0])

                # Split and override train & valid loaders
                X_train, X_valid = [X[i] for i in train_idx], [X[i] for i in valid_idx]
                y_train, y_valid = [y[i] for i in train_idx], [y[i] for i in valid_idx]
                train_dataset_fold, valid_dataset_fold = [], []
                for idx, x in enumerate(X_train): train_dataset_fold.append((x, y_train[idx]))
                for idx, x in enumerate(X_valid): valid_dataset_fold.append((x, y_valid[idx]))
                train_dataloader = DataLoader(train_dataset_fold, batch_size=combination[1], shuffle=True)
                valid_dataloader = DataLoader(valid_dataset_fold, batch_size=combination[1], shuffle=False)
                self.train_dataloader = train_dataloader
                self.valid_dataloader = valid_dataloader
                self.train(predict=False, show_progress=False)
                val_loss_list.append(self.performance_tracker.best_valid_loss)
            mean_val_loss = np.mean(val_loss_list)
            if np.mean(val_loss_list) < best_mean_val_loss:
                best_mean_val_loss = mean_val_loss
                best_combination = combination_idx
            print(np.mean(val_loss_list), best_mean_val_loss, combinations[best_combination])
        print('Best loss, hyperparameters:', best_mean_val_loss, combinations[best_combination])

        # Now train using the valid scaffold and best combination
        # Reinitialize stateful components
        print('Optimizing final model with best hyperparameters')
        self.performance_tracker.reset()
        self._initialize_model(combinations[best_combination][2], combinations[best_combination][3],
                               combinations[best_combination][4],
                               combinations[best_combination][5], combinations[best_combination][6],
                               combinations[best_combination][7], repr_type=self.params['repr_type'])
        self._initialize_optimizer(lr=combinations[best_combination][0])
        train_dataloader = DataLoader(train_dataset, batch_size=combinations[best_combination][1], shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=combinations[best_combination][1], shuffle=False)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.train()

    def submit(self, predictions) -> None:
        self.competition.submit_predictions(
            predictions=predictions,
            prediction_name="ecfp_simple",
            prediction_owner="aehrlich",
            report_url="https://www.example.com",
            description="First test submission"
        )

    def train(self, predict=True, show_progress=True) -> None:
        for epoch in tqdm(range(self.params["epochs"]), disable=not show_progress):
            self.performance_tracker.log({"epoch": epoch})
            self._train_loop(epoch, self.train_dataloader)
            self._valid_loop(epoch, self.valid_dataloader)

            self.performance_tracker.update_early_loss_state()
            if self.performance_tracker.early_stop:
                break
        if predict:
            self.predict()
            self.performance_tracker.save_performance()

    def predict(self) -> list:
        out = []
        for smiles in self.test_dataloader:
            self.performance_tracker.test_pred["SMILES"] = smiles
            out = self.model(smiles)

        self.performance_tracker.test_pred["y"] = torch.flatten(out).tolist()
        return out

    def _train_loop(self, epoch, dataloader):
        self.model.train()
        epoch_loss = 0
        for batch, (smiles, labels) in enumerate(dataloader):
            labels = labels.view(-1, 1).to(self.device)
            out = self.model(smiles)
            loss = self.loss_fn(out, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        self.performance_tracker.log({"train_loss": average_loss})

    def _valid_loop(self, epoch, dataloader):
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


def main(config):
    y_pred = {}
    competition = po.load_competition(f"asap-discovery/{config["task"]}-2025")
    target_cols = competition.target_cols
    for target_col in target_cols:
        performance_tracker = PerformanceTracker(tracking_dir=Path("./models"), id_run=target_col)
        params = config
        params["target_col"] = target_col
        potency = AdmetPotency(params=config, performance_tracker=performance_tracker)
        # potency.train()
        potency.cross_validate()
        y = potency.predict()
        y_pred[target_col] = torch.flatten(y).tolist()


if __name__ == '__main__':
    print("No direct call allowed to this file.")
