"""
This file will contain all of our models.
"""

import sys

import torch
import torch_geometric.utils.smiles as pyg_smiles
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn
from torch_geometric.nn import GAT, GCN, GIN, GraphSAGE, global_add_pool

from src.himp import Himp


def create_repr_model(params: dict) -> nn.Module:
    match params["repr_model"]:
        case "ECFP":
            repr_model = ECFPModel(radius=params["radius"], fpSize=params["out_channels"])
        case "GIN":
            repr_model = GINModel(
                in_channels=params["in_channels"],
                hidden_channels=params["hidden_channels"],
                out_channels=params["out_channels"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                encoding_dim=params["encoding_dim"],
            )
        case "GCN":
            repr_model = GCNModel(
                in_channels=params["in_channels"],
                hidden_channels=params["hidden_channels"],
                out_channels=params["out_channels"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                encoding_dim=params["encoding_dim"],
            )
        case "GAT":
            repr_model = GATModel(
                in_channels=params["in_channels"],
                hidden_channels=params["hidden_channels"],
                out_channels=params["out_channels"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                encoding_dim=params["encoding_dim"],
            )
        case "GraphSAGE":
            repr_model = GraphSAGEModel(
                in_channels=params["in_channels"],
                hidden_channels=params["hidden_channels"],
                out_channels=params["out_channels"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                encoding_dim=params["encoding_dim"],
            )
        case "HIMP":
            repr_model = HIMPModel(
                in_channels=params["in_channels"],
                hidden_channels=params["hidden_channels"],
                out_channels=params["out_channels"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
            )
        case _:
            raise NotImplementedError

    return repr_model


def create_proj_model(params: dict) -> nn.Module:
    return ProjectionHead(
        in_dim=params["out_channels"],
        out_dim=params["out_dim"],
        hidden_dim=params["proj_hidden_dim"],
    )


class PolarisModel(nn.Module):
    def __init__(self, repr_model: nn.Module, proj_model: nn.Module):
        super().__init__()
        self.repr_model = repr_model
        self.proj_model = proj_model

    def forward(self, data):
        h = self.repr_model(data)
        z = self.proj_model(h)
        return z


class GINModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        encoding_dim: int,
    ):
        super().__init__()
        self.encoding_model = CategoricalEncodingModel(embedding_dim=encoding_dim)
        in_channels = self.encoding_model.get_feature_embedding_dim()

        self.model = GIN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.pool = global_add_pool

    def forward(self, data):
        data = self.encoding_model(data)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h_G = self.pool(x=h, batch=data.batch)

        return h_G


class GCNModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        encoding_dim: int,
    ):
        super().__init__()
        self.encoding_model = CategoricalEncodingModel(embedding_dim=encoding_dim)
        in_channels = self.encoding_model.get_feature_embedding_dim()

        self.model = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.pool = global_add_pool

    def forward(self, data):
        data = self.encoding_model(data)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h_G = self.pool(x=h, batch=data.batch)

        return h_G


class GATModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        encoding_dim: int,
    ):
        super().__init__()
        self.encoding_model = CategoricalEncodingModel(embedding_dim=encoding_dim)
        in_channels = self.encoding_model.get_feature_embedding_dim()

        self.model = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.pool = global_add_pool

    def forward(self, data):
        data = self.encoding_model(data)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h_G = self.pool(x=h, batch=data.batch)

        return h_G


class GraphSAGEModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        encoding_dim: int,
    ):
        super().__init__()
        self.encoding_model = CategoricalEncodingModel(embedding_dim=encoding_dim)
        in_channels = self.encoding_model.get_feature_embedding_dim()

        self.model = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.pool = global_add_pool

    def forward(self, data):
        data = self.encoding_model(data)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h_G = self.pool(x=h, batch=data.batch)

        return h_G


class HIMPModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.model = Himp(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, data):
        return self.model(data)


class ECFPModel(nn.Module):
    def __init__(self, radius: int, fpSize: int):
        super().__init__()
        self.fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)

    def forward(self, data):
        mols = [Chem.MolFromSmiles(smiles) for smiles in data.smiles]
        ecfps = [list(ecfp) for ecfp in self.fpgen.GetFingerprints(mols)]
        return torch.tensor(ecfps, dtype=torch.float32)  # could also return as uint


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        return self.projection(data)


class CategoricalEncodingModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.node_embedding = CategoricalEmbeddingModel(
            category_type="node", embedding_dim=embedding_dim
        )
        self.edge_embedding = CategoricalEmbeddingModel(
            category_type="edge", embedding_dim=embedding_dim
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)
        data.edge_attr = self.edge_embedding(data.edge_attr)

        return data

    def get_feature_embedding_dim(self):
        return self.node_embedding.get_node_feature_dim()

    def get_edge_embedding_dim(self):
        return self.edge_embedding.get_edge_feature_dim()


class CategoricalEmbeddingModel(nn.Module):
    """
    Model to embed categorical node or edge features
    """

    def __init__(self, category_type, embedding_dim=8):
        super().__init__()
        if category_type == "node":
            num_categories = self._get_num_node_categories()
        elif category_type == "edge":
            num_categories = self._get_num_edge_categories()
        else:
            print("Invalid category type")
            sys.exit()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories[i], embedding_dim)
                for i in range(len(num_categories))
            ]
        )

    def forward(self, x):
        embedded_vars = [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))]

        return torch.cat(embedded_vars, dim=-1)

    def get_node_feature_dim(self):
        return len(self._get_num_node_categories() * self.embedding_dim)

    def get_edge_feature_dim(self):
        return len(self._get_num_edge_categories() * self.embedding_dim)

    @staticmethod
    def _get_num_node_categories() -> list[int]:
        return [
            len(pyg_smiles.x_map[prop]) for prop in pyg_smiles.x_map
        ]  # [119, 9, 11, 12, 9, 5, 8, 2, 2]

    @staticmethod
    def _get_num_edge_categories() -> list[int]:
        return [len(pyg_smiles.e_map[prop]) for prop in pyg_smiles.e_map]  # [22, 6, 2]
