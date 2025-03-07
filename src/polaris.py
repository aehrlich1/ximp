from multiprocessing import Manager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import PolarisDataset
from src.models import PolarisModel, create_repr_model, create_proj_model
from src.utils import (
    PerformanceTracker,
    scaffold_split,
    make_combinations,
    save_dict_to_csv,
)


class Polaris:
    def __init__(self, params: dict, queue=None):
        self.params: dict = params
        self.queue = queue
        self.performance_tracker = PerformanceTracker(Path("./models"), id_run="x")
        self.device: str = "cpu"
        self.train_polaris = None
        self.test_polaris = None
        self.train_scaffold = None
        self.test_scaffold = None
        self.loss_fn = None
        self.optimizer = None
        self.model = None

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
        skf = StratifiedKFold(
            n_splits=self.params["num_cv_folds"], shuffle=True, random_state=42
        )

        print("Running K-Fold CV...")
        val_loss_list = []

        for fold, (train_idx, valid_idx) in enumerate(skf.split(smiles, y_binned)):
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

        print(f"Validation losses: {val_loss_list}")
        print(f"Average validation loss: {np.mean(val_loss_list)}")

        self.params.update({"mean_val_loss": np.mean(val_loss_list)})
        self.params.update({"patience": self.performance_tracker.patience})
        self.params.update(
            {
                "final_avg_epochs": round(
                    np.mean(self.performance_tracker.early_stop_epoch)
                )
            }
        )

        if self.queue is not None:
            self.queue.put(self.params)

    def train(self, train_dataloader, valid_dataloader) -> None:
        for epoch in tqdm(range(self.params["epochs"])):
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
        for _ in tqdm(range(self.params["final_avg_epochs"])):
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
        self.optimizer = Adam(self.model.parameters(), lr=self.params["lr"])

    def _init_dataset(self):
        root = Path("./data") / "polaris" / self.params["task"]

        log_transform = True if self.params["task"] == "admet" else False

        self.train_polaris = PolarisDataset(
            root=root,
            task=self.params["task"],
            target_task=self.params["target_task"],
            train=True,
            log_transform=log_transform,
            force_reload=False,
        )
        self.test_polaris = PolarisDataset(
            root=root,
            task=self.params["task"],
            target_task=self.params["target_task"],
            train=False,
            log_transform=log_transform,
            force_reload=False,
        )

        self.train_scaffold, self.test_scaffold = scaffold_split(
            dataset=self.train_polaris, test_size=self.params["scaffold_split_val_sz"]
        )

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


class PolarisDispatcher:
    def __init__(self, params: dict) -> None:
        self.params = params

    def run(self):
        torch.set_num_threads(1)
        with Manager() as manager:
            queue = manager.Queue()

            params_list: list[dict] = make_combinations(self.params)

            print(f"Total param count: {len(params_list)}")
            print(f"Using device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")

            for params in params_list:
                self.worker(params, queue)

            result = []
            while not queue.empty():
                result.append(queue.get())

            if isinstance(self.params["repr_model"], list):
                name = "gnn"
            else:
                name = self.params["repr_model"].lower()

            results_path: Path = (
                Path(".") / "results" / f"{self.params['task']}_{name}_results.csv"
            )
            results_path.parent.mkdir(parents=True, exist_ok=True)
            save_dict_to_csv(result, results_path)

    @staticmethod
    def worker(params, queue):
        polaris = Polaris(params, queue)
        polaris.run()

