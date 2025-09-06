import torch
import torch_geometric.utils.smiles as pyg_smiles
from torch import nn


class AtomEncoder(torch.nn.Module):
    """
    Based on implementation from HIMP-GNN
    https://github.com/rusty1s/himp-gnn
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList()
        num_node_categories: list = _get_num_node_categories()

        for i in range(len(num_node_categories)):
            emb = nn.Embedding(num_node_categories[i], embedding_dim)
            self.embeddings.append(emb)

    def forward(self, x):
        out = 0
        for i in range(len(self.embeddings)):
            out += self.embeddings[i](x[:, i])
        return out


class BondEncoder(torch.nn.Module):
    """
    Based on implementation from HIMP-GNN
    https://github.com/rusty1s/himp-gnn
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList()
        num_edge_categories: list = _get_num_edge_categories()

        for i in range(len(num_edge_categories)):
            emb = nn.Embedding(num_edge_categories[i], embedding_dim)
            self.embeddings.append(emb)

    def forward(self, edge_attr):
        out = 0
        for i in range(len(self.embeddings)):
            out += self.embeddings[i](edge_attr[:, i])
        return out


def _get_num_node_categories() -> list[int]:
    return [
        len(pyg_smiles.x_map[prop]) for prop in pyg_smiles.x_map
    ]  # [119, 9, 11, 12, 9, 5, 8, 2, 2]


def _get_num_edge_categories() -> list[int]:
    return [len(pyg_smiles.e_map[prop]) for prop in pyg_smiles.e_map]  # [22, 6, 2]
