import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GINConv, GINEConv
from torch_scatter import scatter

from src.transform import ReducedGraphData


class AtomEncoder(torch.nn.Module):
    """
        Taken from: Matthias Fey, Jan-Gin Yuen, and Frank Weichert. Hierarchical inter-
        message passing for learning on molecular graphs. ArXiv, abs/2006.12179, 2020.

        Github: https://github.com/rusty1s/himp-gnn/blob/master/model.py
    """
    def __init__(self, hidden_channels):
        super(AtomEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(9):
            emb = Embedding(100, hidden_channels)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.embeddings.append(emb)

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

        for i in range(3):
            emb = Embedding(100, hidden_channels)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.embeddings.append(emb)

    def forward(self, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        out = 0
        for i in range(edge_attr.size(1)):
            out += self.embeddings[i](edge_attr[:, i])
        return out

class EHimp(torch.nn.Module):
    """
    Neural network model from the thesis.


    Baed on: Matthias Fey, Jan-Gin Yuen, and Frank Weichert. Hierarchical inter-
    message passing for learning on molecular graphs. ArXiv, abs/2006.12179, 2020.

    Github: https://github.com/rusty1s/himp-gnn/blob/master/model.py

    """
    def __init__(self, hidden_channels, out_channels, num_layers, dropout=0.0,
                 rg_num=1, nums_of_features=[8], device='cpu', use_raw=True, inter_message_passing=True): #TODO Hyperparameters need either be infered or  go in config
        """
        Constructor for NetCustom.

        Parameters:
            - hidden_channels (int): Number of hidden channels in the model.
            - out_channels (int): Number of output channels in the model.
            - num_layers (int): Number of GNN layers in the model.
            - dropout (float): Dropout probability.
            - rg_num (int): Number of reduced graphs.
            - nums_of_features (list): List of numbers of features for each reduced graph.
            - device (str): Device for computation ('cpu' or 'cuda').
            - use_raw (bool): Flag to indicate whether to use raw graph data.
            - inter_message_passing (bool): Flag to enable inter-message passing.

        """
        super(EHimp, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.rg_num = rg_num
        self.use_raw = use_raw
        self.inter_message_passing = inter_message_passing

        # Atom encoder for raw graph data
        self.atom_encoder = AtomEncoder(hidden_channels)

        # Embeddings for reduced graphs
        self.rg_embeddings = ModuleList()
        for i in range(rg_num):
            self.rg_embeddings.append(Embedding(nums_of_features[i], hidden_channels)).to(device)

        # GNN layers for raw graph data
        self.bond_encoders = ModuleList()
        self.atom_convs = ModuleList()
        self.atom_batch_norms = ModuleList()

        for _ in range(num_layers):
            self.bond_encoders.append(BondEncoder(hidden_channels)).to(device)
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.atom_convs.append(GINEConv(nn, train_eps=True)).to(device)
            self.atom_batch_norms.append(BatchNorm1d(hidden_channels)).to(device)

        # GNN layers for reduced graphs
        self.rg_convs = []
        self.rg_batch_norms = []

        for i in range(rg_num):
            convs = ModuleList()
            batch_norms = ModuleList()

            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(),
                    Linear(2 * hidden_channels, hidden_channels),
                )
                convs.append(GINConv(nn, train_eps=True)).to(device)
                batch_norms.append(BatchNorm1d(hidden_channels)).to(device)

            self.rg_convs.append(convs)
            self.rg_batch_norms.append(batch_norms)

        # Linear layers for mapping between raw and reduced graphs
        self.rg2raw_lins = []

        for i in range(rg_num):
            rg2raw_lins = ModuleList()

            for j in range(num_layers):
                rg2raw_lins.append(Linear(hidden_channels, hidden_channels)).to(device)

            self.rg2raw_lins.append(rg2raw_lins)

        # Additional linear layers for mapping between raw and reduced graphs
        if self.inter_message_passing and self.use_raw:
            self.raw2rg_lins = []
            for i in range(rg_num):
                raw2rg_lins = ModuleList()

                for j in range(num_layers):
                    raw2rg_lins.append(Linear(hidden_channels, hidden_channels)).to(device)

                self.raw2rg_lins.append(raw2rg_lins)

        # Final linear layers
        self.atom_lin = Linear(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

        self.rg_lins = ModuleList()
        for i in range(rg_num):
            self.rg_lins.append(Linear(hidden_channels, hidden_channels)).to(device)

    def __collect_rg_from_data(self, data):
        """
        Collect reduced graph data from data object.
        """
        reduced_graphs = []
        idx = 0
        while hasattr(data, f'rg_edge_index_{idx}'):
            rg_data = ReducedGraphData()
            setattr(rg_data, f'rg_edge_index', getattr(data, f'rg_edge_index_{idx}'))
            setattr(rg_data, f'mapping', getattr(data, f'mapping_{idx}'))
            setattr(rg_data, f'rg_num_atoms', getattr(data, f'rg_num_atoms_{idx}'))
            setattr(rg_data, f'rg_atom_features', getattr(data, f'rg_atom_features_{idx}'))
            idx+=1
            #rg_data.to(data.get_device())
            reduced_graphs.append(rg_data)
        return reduced_graphs

    def forward(self, data):
        """
        Forward pass of the model.

        Parameters:
            - data: Graph data for raw graph.
            - reduced_graphs: List of reduced graph data.

        Returns:
            - x: Model output.

        """
        reduced_graphs = self.__collect_rg_from_data(data)
        rg_num = len(reduced_graphs)

        # Atom encoding for raw graph
        x = self.atom_encoder(data.node_feat.squeeze())

        # Embeddings for reduced graphs
        rgs = []
        for i in range(self.rg_num):
            rgs.append(self.rg_embeddings[i](reduced_graphs[i].rg_atom_features.squeeze()))

        # GNN layers for raw graph
        for i in range(self.num_layers):
            if self.use_raw:
                edge_attr = self.bond_encoders[i](data.edge_feat)
                x = self.atom_convs[i](x, data.edge_index, edge_attr)
                x = self.atom_batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
            # GNN layers for reduced graphs
            for j in range(self.rg_num):
                row, col = reduced_graphs[j].mapping
                rg = rgs[j]

                if self.inter_message_passing:
                    rg = rg + F.relu(self.raw2rg_lins[j][i](scatter(x[row], col,
                                    dim=0, dim_size=rg.size(0), reduce='mean')))

                rg = self.rg_convs[j][i](rg, reduced_graphs[j].rg_edge_index)
                rg = self.rg_batch_norms[j][i](rg)
                rg = F.relu(rg)
                rg = F.dropout(rg, self.dropout, training=self.training)

                if self.inter_message_passing:
                    x = x + F.relu(self.rg2raw_lins[j][i](scatter(
                        rg[col], row, dim=0, dim_size=x.size(0),
                        reduce='mean')))

        # Aggregation for raw graph
        if self.use_raw:
            x = scatter(x, data.batch, dim=0, reduce='mean')
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.atom_lin(x)

        # Linear layers for reduced graphs
        for i in range(self.rg_num):
            tree_batch = torch.repeat_interleave(reduced_graphs[i].rg_num_atoms.type(torch.int64))
            rg = rgs[i]
            rg = scatter(rg, tree_batch, dim=0, dim_size=data.y.size(0), reduce='mean')
            rg = F.dropout(rg, self.dropout, training=self.training)
            rg = self.rg_lins[i](rg)
            if self.use_raw:
                x = x + rg
            else:
                x = rg

        # Readout
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)

        return x