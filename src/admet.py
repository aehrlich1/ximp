from pathlib import Path

import polaris as po
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.data import AdmetDataset, AdmetTestDataset
from src.models import AdmetModel
from src.utils import filter_and_extract, PerformanceTracker


class Admet:
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
        self._initialize()

    def _initialize(self):
        self._initialize_loss_fn()
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_data()

    def _initialize_loss_fn(self):
        self.loss_fn = nn.MSELoss()

    def _initialize_model(self):
        self.model = AdmetModel(in_dim=1024, out_dim=1)

    def _initialize_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def _initialize_data(self):
        self.competition = po.load_competition(f"asap-discovery/antiviral-potency-2025")
        data_dir: Path = Path("../data/antiviral-potency-2025")
        self.competition.cache(data_dir)

        train, test = self.competition.get_train_test_split()
        data = filter_and_extract(train, target_col=self.params["target_col"])
        dataset = AdmetDataset(data)
        test_dataset = AdmetTestDataset(test)

        train_dataset, valid_dataset = random_split(dataset=dataset, lengths=[0.9, 0.1])
        train_dataloader = DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=32, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

    def submit(self, predictions) -> None:
        self.competition.submit_predictions(
            predictions=predictions,
            prediction_name="ecfp_simple",
            prediction_owner="aehrlich",
            report_url="https://www.example.com",
            description="First test submission"
        )

    def train(self) -> None:
        for epoch in tqdm(range(self.params["epochs"])):
            self.performance_tracker.log({"epoch": epoch})
            self._train_loop(epoch, self.train_dataloader)
            self._valid_loop(epoch, self.valid_dataloader)

            self.performance_tracker.update_early_loss_state()
            if self.performance_tracker.early_stop:
                break

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
            labels = labels.view(-1, 1)
            out = self.model(smiles)
            loss = self.loss_fn(out, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        self.performance_tracker.log({"train_loss": average_loss})
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}, Train Loss: {average_loss}")

    def _valid_loop(self, epoch, dataloader):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for smiles, labels in dataloader:
                labels = labels.view(-1, 1)
                out = self.model(smiles)
                loss = self.loss_fn(out, labels)
                epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        self.performance_tracker.log({"valid_loss": average_loss})
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}, Valid Loss: {average_loss}")


def main():
    y_pred = {}

    competition = po.load_competition(f"asap-discovery/antiviral-admet-2025")
    target_cols = competition.target_cols

    for target_col in target_cols:
        performance_tracker = PerformanceTracker(tracking_dir=Path("../models"), id_run=target_col)
        admet = Admet(params={"epochs": 200, "target_col": target_col}, performance_tracker=performance_tracker)
        admet.train()
        y = admet.predict()
        y_pred[target_col] = torch.flatten(y).tolist()

    print(y_pred)
    competition.submit_predictions(
        predictions = y_pred,
        prediction_name = "ecfp_simple",
        prediction_owner = "aehrlich",
        report_url = "https://www.example.com",
        description = "First test submission"
    )

def main_potency():
    y_pred = {}

    competition = po.load_competition(f"asap-discovery/antiviral-potency-2025")
    target_cols = competition.target_cols
    for target_col in target_cols:
        performance_tracker = PerformanceTracker(tracking_dir=Path("../models"), id_run=target_col)
        potency = Admet(params={"epochs": 200, "target_col": target_col}, performance_tracker=performance_tracker)
        potency.train()
        y = potency.predict()
        y_pred[target_col] = torch.flatten(y).tolist()


    print(y_pred)
    competition.submit_predictions(
        predictions = y_pred,
        prediction_name = "ecfp_simple",
        prediction_owner = "aehrlich",
        report_url = "https://www.example.com",
        description = "First test submission"
    )

if __name__ == '__main__':
    main_potency()
