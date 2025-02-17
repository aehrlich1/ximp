"""
This file will contain all of our models.
"""
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GIN
from src.himp import HimpModel


def create_repr_model(params: dict) -> nn.Module:
    match params['repr_model']:
        case "HIMP":
            repr_model = HimpModel(
                hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                inter_message_passing=params['inter_message_passing'],
            )
        case "ECFP":
            repr_model = ECFPModel(radius=params['radius'], fpSize=params['out_channels'])
        case "GIN":
            repr_model = GINModel(
                in_channels=params['in_channels'],
                hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
            )
        case "GCN":
            repr_model = GCNModel(
                in_channels=params['in_channels'],
                hidden_channels=params['hidden_channels'],
                out_channels=params['out_channels'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
            )
        case _:
            raise NotImplementedError

    return repr_model


def create_proj_model(params: dict, hidden_dim: int = 64) -> nn.Module:
    return ProjectionHead(in_dim=params["out_channels"], out_dim=params["out_dim"], hidden_dim=hidden_dim)


class PolarisModel(nn.Module):
    def __init__(self, repr_model: nn.Module, proj_model: nn.Module):
        super().__init__()
        self.repr_model = repr_model
        self.proj_model = proj_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data):
        h = self.repr_model(data)
        z = self.proj_model(h)
        return z.to(self.device)


class GINModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.model = GIN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                         num_layers=num_layers, dropout=dropout)
        self.pool = global_add_pool

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h_G = self.pool(x=h, batch=data.batch)

        return h_G


class GCNModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.model = GIN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                         num_layers=num_layers, dropout=dropout)
        self.pool = global_add_pool

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h_G = self.pool(x=h, batch=data.batch)

        return h_G


class ECFPModel(nn.Module):
    def __init__(self, radius=2, fpSize=1024):
        super().__init__()
        self.fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)

    def forward(self, data):
        mols = [Chem.MolFromSmiles(smiles) for smiles in data.smiles]
        ecfps = [list(ecfp) for ecfp in self.fpgen.GetFingerprints(mols)]
        return torch.tensor(ecfps, dtype=torch.float32)  # could also return as uint


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        ).to(self.device)

    def forward(self, data):
        return self.projection(data.to(self.device))
