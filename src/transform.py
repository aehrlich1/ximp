# Based on: https://github.com/rusty1s/himp-gnn/blob/master/transform.py
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch_geometric.data import Data
from torch_geometric.utils import tree_decomposition

bonds = [bond for bond in BondType.__dict__.values() if isinstance(bond, BondType)]


def mol_from_data(data):
    mol = Chem.RWMol()

    x = data.x if data.x.dim() == 1 else data.x[:, 0]
    for z in x.tolist():
        mol.AddAtom(Chem.Atom(z))

    row, col = data.edge_index
    mask = row < col
    row, col = row[mask].tolist(), col[mask].tolist()

    bond_type = data.edge_attr
    bond_type = bond_type if bond_type.dim() == 1 else bond_type[:, 0]
    bond_type = bond_type[mask].tolist()

    for i, j, bond in zip(row, col, bond_type):
        # assert bond >= 1 and bond <= 4
        mol.AddBond(i, j, bonds[bond - 1])

    return mol.GetMol()


class JunctionTreeData(Data):
    def __inc__(self, key, item, *args):
        if key == "tree_edge_index":
            return self.x_clique.size(0)
        elif key == "atom2clique_index":
            return torch.tensor([[self.x.size(0)], [self.x_clique.size(0)]])
        else:
            return super(JunctionTreeData, self).__inc__(key, item, *args)


class JunctionTree(object):
    def __call__(self, data):
        mol = mol_from_data(data)
        out = tree_decomposition(mol, return_vocab=True)
        tree_edge_index, atom2clique_index, num_cliques, x_clique = out

        data = JunctionTreeData(**{k: v for k, v in data})

        data.tree_edge_index = tree_edge_index
        data.atom2clique_index = atom2clique_index
        data.num_cliques = num_cliques
        data.x_clique = x_clique

        return data


class OGBTransform(object):
    # OGB saves atom and bond types zero-index based. We need to revert that.
    def __call__(self, data):
        data.x[:, 0] += 1
        data.edge_attr[:, 0] += 1
        return data

class FeatureTree(object):
    def __call__(self, data):
        mol = mol_from_data(data)
        out = tree_decomposition(mol, return_vocab=True)

        data = ReducedGraphData(**{k: v for k, v in data})
        data.node_feat = data.x # Compatibility w/ EHimp TODO adhere to naming convention
        data.edge_feat = data.edge_attr # Compatibility EHimp
        data.rg_edge_index_0, data.mapping_0, data.rg_num_atoms_0, data.rg_atom_features_0 = out
        data.raw_num_atoms_0 = data.x.size(0)

        resolution=2 #TODO make config param
        for i in range(0, resolution):
            data = getFeatureTreeWithLowerResolution(data, i+1) #i here is just for naming the attributes

        return data
class ReducedGraphData(Data):
    """
    Custom data class for storing information related to the Reduced Graph.

    Attributes:
        - rg_edge_index (Tensor): Edge indices of the Reduced Graph.
        - mapping (Tensor): Mapping information between the raw graph and the Reduced Graph.
        - rg_num_atoms (Tensor): Number of atoms in the Reduced Graph.
        - raw_num_atoms (int): Number of atoms in the raw graph.

    Methods:
        - __cat_dim__(self, key, value, *args, **kwargs): Custom implementation for concatenation dimension.
        - __inc__(self, key, value, *args, **kwargs): Custom implementation for incremental value.

    """
    def __cat_dim__(self, key, value, *args, **kwargs):
        if any(word in key for word in ['edge_index', 'rg_edge_index', 'mapping']):
        #if key in ['edge_index', 'rg_edge_index', 'mapping']:
            return 1
        else:
            return 0

    def __inc__(self, key, value, *args, **kwargs):
        idx = key.split('_')[-1]
        if key == 'edge_index':
            return getattr(self, f'raw_num_atoms_0') #self.raw_num_atoms, always the same of teh original graph
        elif 'rg_edge_index' in key:
            return getattr(self, f'rg_num_atoms_{idx}')
        elif 'mapping' in key:
            return torch.tensor([[torch.sum(getattr(self, f'raw_num_atoms_{idx}'))], [getattr(self, f'rg_num_atoms_{idx}')]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


def getFeatureTreeData(molecule, num_of_nodes, resolution=0): # Construction reminiscent of HIMP junction trees.
    """
    Get data for the Feature Tree (FT) from a given molecule with a specified resolution.

    Parameters:
        - molecule (Chem.rdchem.Mol): RDKit Mol object representing a molecule.
        - num_of_nodes (int): Number of nodes in the raw graph.
        - resolution (int): Resolution level for the Feature Tree. Use 0 for standard FT.

    Returns:
        - data (ReducedGraphData): Data for the FT in the form of ReducedGraphData, including features and mapping.
    """

    out = tree_decomposition(molecule, return_vocab=True)

    data = ReducedGraphData()
    data.rg_edge_index, data.mapping, data.rg_num_atoms, data.rg_atom_features = out
    data.raw_num_atoms = num_of_nodes

    # Recursively get FT with lower resolution if specified
    for _ in range(resolution):
        data = getFeatureTreeWithLowerResolution(data)

    return data


def getFeatureTreeWithLowerResolution(tree, resolution=1):

    rg_edge_index = getattr(tree, f'rg_edge_index_{resolution-1}')
    rg_num_atoms  = getattr(tree, f'rg_num_atoms_{resolution-1}')
    raw_num_atoms = getattr(tree, f'raw_num_atoms_{resolution-1}')
    mapping = getattr(tree, f'mapping_{resolution - 1}')
    rg_atom_features = getattr(tree, f'rg_atom_features_{resolution - 1}')

    unique_values, counts = torch.unique(rg_edge_index[0], return_counts=True)

    leaf_idxs = unique_values[counts == 1]  # Leaf nodes
    non_leaf_idxs = unique_values[counts > 1]  # Inner nodes
    if rg_edge_index.shape[1] == 2:  # in case of one edge:
        leaf_idxs = leaf_idxs[1:]
        non_leaf_idxs = torch.tensor([leaf_idxs[0]])
    unconnected = np.setdiff1d(np.arange(rg_num_atoms), unique_values)

    # Atrributes of the resulting tree
    new_rg_edge_index = torch.clone(rg_edge_index)
    new_mapping = torch.clone(mapping)
    new_rg_atom_features = torch.clone(rg_atom_features)
    new_rg_num_atoms = rg_num_atoms - len(leaf_idxs)

    non_leaf_edges = torch.logical_and(torch.isin(rg_edge_index[0], non_leaf_idxs), torch.isin(rg_edge_index[1], non_leaf_idxs)) # Edges that are not connecting leaf nodes
    new_rg_edge_index = rg_edge_index[:, non_leaf_edges]

    idx_reduction = torch.zeros(rg_num_atoms, dtype=torch.int64) # Array that maps the gap between the index of a node in the original and new trees
    parents = rg_edge_index[1, torch.isin(rg_edge_index[0], leaf_idxs)] # Parents of leaves

    for leaf, parent in zip(leaf_idxs, parents):
        idx_reduction[leaf:] += 1

        new_mapping[1, new_mapping[1] == leaf] = parent  # Map nodes that are mapped to the leaf to its parent

        if new_rg_atom_features[leaf] < new_rg_atom_features[parent]:  # Change the feature attribute it needed
            new_rg_atom_features[parent] = new_rg_atom_features[leaf].to(torch.int64)

    # Delete multiple occurences
    new_mapping, _ = torch.unique(new_mapping, dim=1, return_inverse=True)

    new_rg_atom_features = concatenated_array = new_rg_atom_features[np.concatenate((non_leaf_idxs, unconnected))]

    #Indexing
    new_rg_edge_index -= idx_reduction[new_rg_edge_index]
    new_mapping[1] -= idx_reduction[new_mapping[1]]

    # Create new data point
    reduced_tree = ReducedGraphData(**{k: v for k, v in tree})

    setattr(reduced_tree, f'rg_edge_index_{resolution}', new_rg_edge_index)
    setattr(reduced_tree, f'mapping_{resolution}', new_mapping)
    setattr(reduced_tree, f'rg_num_atoms_{resolution}', new_rg_num_atoms)
    setattr(reduced_tree, f'rg_atom_features_{resolution}', new_rg_atom_features)
    setattr(reduced_tree, f'raw_num_atoms_{resolution}', raw_num_atoms)
    return reduced_tree

#reduced_tree = geetFeatureTreeWithLowerResolution(getFeatureTreeData(mols[i], 0))

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


