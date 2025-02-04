"""
This file will contain all of our models.
"""
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem
# Needed for HIMP
import torch.nn.functional as F
from torch.nn import Embedding, ModuleList
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles
from torch_scatter import scatter
from torch_geometric.nn import GINConv, GINEConv, PairNorm

from src.transform import OGBTransform, JunctionTree


class ExtendedConnectivityFingerprintModel:
    def __init__(self, latent_dim=1024):
        self.fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=latent_dim) #TODO: Put in the config

    def __call__(self, dataloader):
        mols = [Chem.MolFromSmiles(smiles) for smiles in dataloader]
        ecfps = [list(ecfp) for ecfp in self.fpgen.GetFingerprints(mols)]
        return ecfps


class EcfpModel(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.ecfp_model = ExtendedConnectivityFingerprintModel(latent_dim)
        #self.linear = nn.Linear(1024, latent_dim)

    def forward(self, data):
        ecfps = self.ecfp_model(data)
        ecfps = torch.tensor(ecfps, dtype=torch.float32)

        return ecfps


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
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

class AdmetPotencyModel(nn.Module):
    def __init__(self, latent_dim, projector_dim, repr_himp_hidden_dim=9, rep_himp_num_layers= 2, rep_himp_dropout=0.5, rep_himp_inter=True, repr_type ='HIMP'):
       #print(latent_dim, projector_dim, flush=True)
        #exit(-1)
        super().__init__()
        self.repr_model = None
        if repr_type == 'HIMP':
            self.repr_model = HIMPModel(hidden_channels=repr_himp_hidden_dim,
                                        out_channels=latent_dim,
                                        num_layers=rep_himp_num_layers,
                                        dropout=rep_himp_dropout,
                                        inter_message_passing=rep_himp_inter
                                        )
        elif repr_type == 'ECFP':
            self.repr_model = EcfpModel(latent_dim=latent_dim) # Produces outputs in latent dim
        self.proj_model = ProjectionHead(in_dim=latent_dim, out_dim=1, hidden_dim=projector_dim) #in_dim

    def forward(self, data):
        h = self.repr_model(data)
        z = self.proj_model(h)
        return z


class AtomEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(AtomEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(9):  # TODO: From config or infer
            self.embeddings.append(Embedding(100, hidden_channels)) # TODO From config or infer

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out


class BondEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BondEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(3): # TODO: From config or infer
            self.embeddings.append(Embedding(
                100  # 6, for testing
                , hidden_channels)) # TODO From config or infer

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        out = 0
        for i in range(edge_attr.size(1)):
            # print(out, edge_attr.shape, self.embeddings[i], i, edge_attr.size(1), flush=True)
            out += self.embeddings[i](edge_attr[:, i])
        return out


class HIMPModel(torch.nn.Module):
    #https://arxiv.org/abs/2006.12179, published @ ICML2020
    def __init__(self, hidden_channels, out_channels, num_layers, dropout=0.0,
                 inter_message_passing=True):
        super(HIMPModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.inter_message_passing = inter_message_passing
        #self.clique_norm = PairNorm(scale_individually=True) # Added

        self.atom_encoder = AtomEncoder(hidden_channels)
        self.clique_encoder = Embedding(4, hidden_channels)

        self.bond_encoders = ModuleList()
        self.atom_convs = ModuleList()
        self.atom_batch_norms = ModuleList()

        for _ in range(num_layers):
            self.bond_encoders.append(BondEncoder(hidden_channels))
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.atom_convs.append(GINEConv(nn, train_eps=True))
            self.atom_batch_norms.append(BatchNorm1d(hidden_channels))

        self.clique_convs = ModuleList()
        self.clique_batch_norms = ModuleList()

        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.clique_convs.append(GINConv(nn, train_eps=True))
            self.clique_batch_norms.append(BatchNorm1d(hidden_channels))

        self.atom2clique_lins = ModuleList()
        self.clique2atom_lins = ModuleList()

        for _ in range(num_layers):
            self.atom2clique_lins.append(
                Linear(hidden_channels, hidden_channels))
            self.clique2atom_lins.append(
                Linear(hidden_channels, hidden_channels))

        self.atom_lin = Linear(hidden_channels, hidden_channels)
        self.clique_lin = Linear(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        self.clique_encoder.reset_parameters()

        for emb, conv, batch_norm in zip(self.bond_encoders, self.atom_convs,
                                         self.atom_batch_norms):
            emb.reset_parameters()
            conv.reset_parameters()
            batch_norm.reset_parameters()

        for conv, batch_norm in zip(self.clique_convs,
                                    self.clique_batch_norms):
            conv.reset_parameters()
            batch_norm.reset_parameters()

        for lin1, lin2 in zip(self.atom2clique_lins, self.clique2atom_lins):
            lin1.reset_parameters()
            lin2.reset_parameters()

        self.atom_lin.reset_parameters()
        self.clique_lin.reset_parameters()
        self.lin.reset_parameters()

    def __pre_forward(self, data):
        data = [OGBTransform()(JunctionTree()(from_smiles(x[0]))) for x in data]
        return Batch.from_data_list(data)

    def forward(self, data):
        data = self.__pre_forward(data)
        x = self.atom_encoder(data.x.squeeze())

        if self.inter_message_passing:
            x_clique = self.clique_encoder(data.x_clique.squeeze())

        for i in range(self.num_layers):
            edge_attr = self.bond_encoders[i](data.edge_attr)
            x = self.atom_convs[i](x, data.edge_index, edge_attr)
            x = self.atom_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.inter_message_passing:
                row, col = data.atom2clique_index

                x_clique = x_clique + F.relu(self.atom2clique_lins[i](scatter(
                    x[row], col, dim=0, dim_size=x_clique.size(0),
                    reduce='mean')))

                x_clique = self.clique_convs[i](x_clique, data.tree_edge_index)
                x_clique = self.clique_batch_norms[i](x_clique)
                x_clique = F.relu(x_clique)
                x_clique = F.dropout(x_clique, self.dropout,
                                     training=self.training)

                x = x + F.relu(self.clique2atom_lins[i](scatter(
                    x_clique[col], row, dim=0, dim_size=x.size(0),
                    reduce='mean')))

        x = scatter(x, data.batch, dim=0, reduce='mean')
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_lin(x)

        if self.inter_message_passing:
            tree_batch = torch.repeat_interleave(data.num_cliques)
            x_clique = scatter(x_clique, tree_batch, dim=0, dim_size=x.size(0),
                               reduce='mean')
            x_clique = F.dropout(x_clique, self.dropout,
                                 training=self.training)
            x_clique = self.clique_lin(x_clique)
            #x_clique = self.clique_norm(x_clique) #Added
            x = x + x_clique

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        #x = self.clique_norm(x)
        return x