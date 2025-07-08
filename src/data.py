"""
This file will take care of all data related aspects.
"""

import atexit
import csv
import os
import shutil
import socket
import time
import uuid
import warnings
from pathlib import Path
from typing import List

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import from_smiles

from src.transform import JunctionTree, ReducedGraph
from src.utils import scaffold_split

# Ignore FutureWarnings from torch.load about weightsOnly bool != True
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)


class MoleculeNetDataset:
    def __init__(
        self,
        root,
        target_task,
        force_reload=False,
        use_erg=False,
        use_ft=False,
        ft_resolution=0,
    ):
        self.root = root
        self.target_task = target_task
        self.force_reload = force_reload
        self.junction_tree = JunctionTree()
        self.reduced_graph = ReducedGraph(
            use_erg=use_erg, use_ft=use_ft, ft_resolution=ft_resolution
        )
        self.use_himp_preprocessing = not (use_ft or use_erg)

    def _pre_transform(
        self, data
    ):  # TODO: rename from pre_transform if we use it as transform below?
        if self.use_himp_preprocessing:
            data = self.junction_tree(data)  # HIMP Graph
        else:
            data = self.reduced_graph(data)  # Extended IMP Graphs

        return data

    def create_dataset(self):
        return MoleculeNet(
            root=self.root,
            name=self.target_task,
            transform=self._pre_transform,
            force_reload=self.force_reload,
        )


class PolarisDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        task: str,
        target_task: str,
        train=True,
        force_reload=True,
        log_transform=True,
        use_erg=False,
        use_ft=False,
        ft_resolution=0,
    ):
        self.target_task = target_task
        self.train = train
        self.force_reload = force_reload
        self.log_transform = log_transform
        self.junctionTree = JunctionTree()
        self.use_himp_preprocessing = not (use_ft or use_erg)
        self.reducedGraph = ReducedGraph(
            use_erg=use_erg, use_ft=use_ft, ft_resolution=ft_resolution
        )

        if task == "admet":
            self.target_col = self._admet_target_to_col_mapping(target_task)
        elif task == "potency":
            self.target_col = self._potency_target_to_col_mapping(target_task)
        else:
            raise ValueError(f"Unknown task: {task}")

        # TODO Would be smarter to generate all combinations only once.
        # Easiest solution would be to just do it OTF.
        # Create unique file names for processed files
        self._uniq = f"{socket.gethostname()}_{os.getpid()}_{str(int(time.time()))}_{uuid.uuid4().hex[:8]}"  # 1:2^(8*8) chance of collision for 8 byte, up to 1:2^(8*32) possible, per process per machine per second
        self._processed_file_names: List[str] = [
            f"train_{self.target_col}_{self._uniq}.pt",
            f"test_{self._uniq}.pt",
        ]
        # Register the same cleanup routine for both __del__ and atexit
        atexit.register(self._cleanup)

        super().__init__(root, force_reload=force_reload)
        self.load(self.processed_paths[0] if train else self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ["train_polaris.csv", "test_polaris.csv"]

    @property
    def processed_dir(self) -> str:
        """
        Every dataset instance gets its *own* directory, eg:
        <root>/processed/<unique_tag>/
        This isolates also PyG’s internal cache files pre_filter.pt / pre_transform.pt.
        """
        return os.path.join(self.root, "processed", self._uniq)

    @property
    def processed_file_names(self):
        # return [f"train_{self.target_col}.pt", "test.pt"]
        return self._processed_file_names

    def process(self):
        self.process_train() if self.train else self.process_test()

    def process_train(self):
        data_list: list[Data] = []
        with open(self.raw_paths[0], "r") as file:
            lines = csv.reader(file)
            next(lines)  # skip header

            for line in lines:
                smiles = line[0]
                label = line[self.target_col]
                if len(label) == 0:
                    continue

                y = torch.tensor(float(label), dtype=torch.float).view(-1, 1)

                if self.log_transform and self.target_task != "LogD":
                    y = torch.log10(y)
                    if y.isinf():
                        y = torch.zeros_like(y)

                data = from_smiles(smiles)
                data.y = y

                if self.use_himp_preprocessing:
                    data = self.junctionTree(data)  # HIMP Graph
                else:
                    data = self.reducedGraph(data)  # Extended HIMP Graphs

                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def process_test(self):
        data_list: list[Data] = []
        with open(self.raw_paths[1], "r") as file:
            lines = csv.reader(file)
            next(lines)  # skip header

            for line in lines:
                smiles = line[0]
                data = from_smiles(smiles)
                data = self.junctionTree(data)
                data_list.append(data)

        self.save(data_list, self.processed_paths[1])

    # Cleanup routine
    def _cleanup_processed_files(self):
        for path in self.processed_paths:
            try:
                pass
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass  # ignore permissions or race conditions

    def _cleanup_processed_dir(self):
        try:
            shutil.rmtree(self.processed_dir, ignore_errors=True)
        except Exception:
            pass  # best-effort; ignore races or permissions

    def _cleanup(self):
        self._cleanup_processed_files()  # TODO: decided whether this redundancy should remain
        self._cleanup_processed_dir()

    def __del__(self):
        # Guaranteed attempt to delete (atexit handles interpreter shutdown)
        self._cleanup_processed_files()
        self._cleanup()

    @staticmethod
    def _admet_target_to_col_mapping(target_task: str) -> int:
        match target_task:
            case "MLM":
                return 1
            case "HLM":
                return 2
            case "KSOL":
                return 3
            case "LogD":
                return 4
            case "MDR1-MDCKII":
                return 5
            case _:
                raise ValueError(f"Unknown target task: {target_task}")

    @staticmethod
    def _potency_target_to_col_mapping(target_task: str) -> int:
        match target_task:
            case "pIC50 (MERS-CoV Mpro)":
                return 1
            case "pIC50 (SARS-CoV-2 Mpro)":
                return 2
            case _:
                raise ValueError(f"Unknown target task: {target_task}")


if __name__ == "__main__":
    dataset = PolarisDataset(
        root="../data/polaris/admet", task="admet", target_task="MDR1-MDCKII", train=True
    )
    train_dataset, valid_dataset = scaffold_split(dataset)
    print(len(train_dataset), len(valid_dataset))
    print(dataset)
