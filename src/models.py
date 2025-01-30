"""
This file will contain all of our models.
"""
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem


class ExtendedConnectivityFingerprintModel:
    def __init__(self):
        self.fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)

    def __call__(self, dataloader):
        mols = [Chem.MolFromSmiles(smiles) for smiles in dataloader]
        ecfps = [list(ecfp) for ecfp in self.fpgen.GetFingerprints(mols)]
        return ecfps


class EcfpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecfp_model = ExtendedConnectivityFingerprintModel()
        self.linear = nn.Linear(1024, 128)

    def forward(self, data):
        ecfps = self.ecfp_model(data)
        ecfps = torch.tensor(ecfps, dtype=torch.float32)

        return ecfps


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, data):
        return self.projection(data)

class AdmetModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.repr_model = EcfpModel()
        self.proj_model = ProjectionHead(in_dim, out_dim)

    def forward(self, data):
        h = self.repr_model(data)
        z = self.proj_model(h)

        return z
