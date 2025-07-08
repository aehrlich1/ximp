# Based on: https://github.com/rusty1s/himp-gnn/blob/master/transform.py
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdReducedGraphs import GenerateMolExtendedReducedGraph
from rdkit.Chem.rdchem import BondType
from torch_geometric.data import Data
from torch_geometric.utils import tree_decomposition

bonds = [bond for bond in BondType.__dict__.values() if isinstance(bond, BondType)]


def mol_from_data(data):
    """
    DEPRECATED - NOT IN USE ANYMORE
    Converts a PyG Data object to an RDKit Mol object.
    TBD: Remove this function as there exists a PyG variant: https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/utils/smiles.html
    """
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


class ReducedGraph(object):
    def __init__(self, use_erg, use_ft, ft_resolution):
        self.use_erg = use_erg
        self.use_ft = use_ft
        self.ft_resolution = ft_resolution

    def __call__(self, data):
        offset = 0
        data = ReducedGraphData(**{k: v for k, v in data})
        data.node_feat = (
            data.x
        )  # Compatibility w/ EHimp TODO change EHIMP to adherence to naming convention
        data.edge_feat = data.edge_attr  # Compatibility EHimp

        if self.use_ft:
            mol = Chem.MolFromSmiles(data.smiles)
            out = tree_decomposition(mol, return_vocab=True)
            data.rg_edge_index_0, data.mapping_0, data.rg_num_atoms_0, data.rg_atom_features_0 = (
                out  # TODO base case should also be encapsulated by addFeatureTreeWithLowerResolution
            )
            data.raw_num_atoms_0 = data.x.size(0)
            for i in range(1, self.ft_resolution):
                data = addFeatureTreeWithLowerResolution(
                    data, i
                )  # The i here is just for naming the attributes
            offset = self.ft_resolution
        if self.use_erg:
            # Generate ErG fingerprint
            mol = Chem.MolFromSmiles(data.smiles)
            erg = getErGData(
                mol, data.x.size(0)
            )  # TODO standardize so it adds graphs just as addFeatureTreeWithLowerResolution
            setattr(data, f"rg_edge_index_{offset}", erg.rg_edge_index)
            setattr(data, f"mapping_{offset}", erg.mapping)
            setattr(data, f"rg_num_atoms_{offset}", erg.rg_num_atoms)
            setattr(data, f"rg_atom_features_{offset}", erg.rg_atom_features)
            setattr(data, f"raw_num_atoms_{offset}", data.x.size(0))
        # print(data, flush=True)
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
        if any(word in key for word in ["edge_index", "rg_edge_index", "mapping"]):
            # if key in ['edge_index', 'rg_edge_index', 'mapping']:
            return 1
        else:
            return 0

    def __inc__(self, key, value, *args, **kwargs):
        idx = key.split("_")[-1]
        if key == "edge_index":
            return getattr(
                self, f"raw_num_atoms_0"
            )  # self.raw_num_atoms, always the same of teh original graph
        elif "rg_edge_index" in key:
            return getattr(self, f"rg_num_atoms_{idx}")
        elif "mapping" in key:
            # return torch.tensor([[torch.sum(getattr(self, f'raw_num_atoms_{idx}'))], [getattr(self, f'rg_num_atoms_{idx}')]])
            x = torch.tensor(getattr(self, f"raw_num_atoms_{idx}"))
            y = torch.tensor(getattr(self, f"rg_num_atoms_{idx}"))
            return torch.tensor([[torch.sum(x)], [y]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


def addFeatureTreeWithLowerResolution(tree, resolution=1):
    rg_edge_index = getattr(tree, f"rg_edge_index_{resolution - 1}")
    rg_num_atoms = getattr(tree, f"rg_num_atoms_{resolution - 1}")
    raw_num_atoms = getattr(tree, f"raw_num_atoms_{resolution - 1}")
    mapping = getattr(tree, f"mapping_{resolution - 1}")
    rg_atom_features = getattr(tree, f"rg_atom_features_{resolution - 1}")

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
    if new_rg_num_atoms >= 1:  # Prevents graphs with 0 nodes
        non_leaf_edges = torch.logical_and(
            torch.isin(rg_edge_index[0], non_leaf_idxs), torch.isin(rg_edge_index[1], non_leaf_idxs)
        )  # Edges that are not connecting leaf nodes
        new_rg_edge_index = rg_edge_index[:, non_leaf_edges]

        idx_reduction = torch.zeros(
            rg_num_atoms, dtype=torch.int64
        )  # Array that maps the gap between the index of a node in the original and new trees
        parents = rg_edge_index[1, torch.isin(rg_edge_index[0], leaf_idxs)]  # Parents of leaves

        for leaf, parent in zip(leaf_idxs, parents):
            idx_reduction[leaf:] += 1

            new_mapping[1, new_mapping[1] == leaf] = (
                parent  # Map nodes that are mapped to the leaf to its parent
            )

            # Original
            if (
                new_rg_atom_features[leaf] < new_rg_atom_features[parent]
            ):  # Change the feature attribute it needed
                new_rg_atom_features[parent] = new_rg_atom_features[leaf].to(torch.int64)

            # Hashing (summation not meaningful as we use indices of embeddings dictionary)
            # print(new_rg_atom_features, leaf, parent, flush=True)
            # def hash_pairwise(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
            #    p1, p2 = 31, 77 # Primes, ideally chosen st they are coprime with k (i.e. they hsare no commong factor but 1 with k) - gives best distribution.
            #    #p1, p2 = 31, 77 (but also 3 and 7) would be coprime to 100
            #    return torch.abs(a * p1 + b * p2) % k

            # new_rg_atom_features[parent] = hash_pairwise(new_rg_atom_features[leaf], new_rg_atom_features[parent], 1000).to(torch.int64)

            # exit(-1)
        # Delete multiple occurences
        new_mapping, _ = torch.unique(new_mapping, dim=1, return_inverse=True)

        new_rg_atom_features = new_rg_atom_features[np.concatenate((non_leaf_idxs, unconnected))]

        # Indexing
        new_rg_edge_index -= idx_reduction[new_rg_edge_index]
        new_mapping[1] -= idx_reduction[new_mapping[1]]
        # print(new_rg_edge_index.shape, new_rg_num_atoms,
        #      rg_edge_index.shape, rg_num_atoms,
        #      new_mapping.shape, mapping.shape,
        #      flush=True)
        # Create new data point
        reduced_tree = ReducedGraphData(**{k: v for k, v in tree})

        setattr(reduced_tree, f"rg_edge_index_{resolution}", new_rg_edge_index)
        setattr(reduced_tree, f"mapping_{resolution}", new_mapping)
        setattr(reduced_tree, f"rg_num_atoms_{resolution}", new_rg_num_atoms)
        setattr(reduced_tree, f"rg_atom_features_{resolution}", new_rg_atom_features)
        setattr(reduced_tree, f"raw_num_atoms_{resolution}", raw_num_atoms)
    else:
        reduced_tree = ReducedGraphData(**{k: v for k, v in tree})

        setattr(reduced_tree, f"rg_edge_index_{resolution}", torch.clone(rg_edge_index))
        setattr(reduced_tree, f"mapping_{resolution}", torch.clone(mapping))
        setattr(reduced_tree, f"rg_num_atoms_{resolution}", rg_num_atoms)
        setattr(reduced_tree, f"rg_atom_features_{resolution}", torch.clone(rg_atom_features))
        setattr(reduced_tree, f"raw_num_atoms_{resolution}", raw_num_atoms)
    return reduced_tree


# reduced_tree = geetFeatureTreeWithLowerResolution(getFeatureTreeData(mols[i], 0))


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def moleculeAtomsProperties(molecule):
    """
    Extracts atom properties from a given RDKit Mol object using SMARTS patterns.

    Parameters:
        - molecule (Chem.rdchem.Mol): RDKit Mol object representing a molecule.

    Returns:
        - properties (numpy.ndarray): Array containing tuples representing atom properties.
          Each tuple indicates the atom indices that match specific chemical patterns.
          The order of patterns corresponds to the following:
          0: Donor atoms
          1: Acceptor atoms
          2: Positively charged atoms
          3: Negatively charged atoms
          4: Hydrophobic atoms
    """

    # Define SMARTS patterns for different atom properties
    donor_pattern = Chem.MolFromSmarts(r"[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
    acceptor_pattern = Chem.MolFromSmarts(
        r"[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O;H0;v2]),$([O,S;v1;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o;+0;!$([o]:n);!$([o]:c:n)])]"
    )
    positive_pattern = Chem.MolFromSmarts(
        r"[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]"
    )
    negative_pattern = Chem.MolFromSmarts(r"[$([C,S](=[O,S,P])-[O;H1,-1])]")
    hydrophobic_pattern = Chem.MolFromSmarts(r"[$([C;D3,D4](-[CH3])-[CH3]),$([S;D2](-C)-C)]")

    # Array to store atom properties
    properties = np.empty(5, dtype=tuple)

    # List of SMARTS patterns
    atom_property_patterns = [
        donor_pattern,
        acceptor_pattern,
        positive_pattern,
        negative_pattern,
        hydrophobic_pattern,
    ]

    # Check if the atom matches any of the specified patterns
    for i, pattern in enumerate(atom_property_patterns):
        properties[i] = molecule.GetSubstructMatches(pattern)

    return properties


def findRecognizedRings(mol):
    """
    Finds rings in a molecule with a ring size less than 8.

    Parameters:
        - mol (Chem.rdchem.Mol): RDKit Mol object representing a molecule.

    Returns:
        - recognized_rings (list): List of atom indices representing rings in the molecule.
          Only rings with a size (number of atoms) less than 8 are included in the result.
    """
    recognized_rings = [ring for ring in mol.GetRingInfo().AtomRings() if len(ring) < 8]
    return recognized_rings


def constructErgFromFp(erg_fp):
    """
    Construct an ErG (Extended Graph) from a given fingerprint.

    Parameters:
        - erg_fp (Chem.AllChem.GetErGFingerprint): ErG fingerprint obtained using RDKit.

    Returns:
        - atom_features (numpy.ndarray): Array of atom features:
            - 0: No feature
            - [1, 2, 3, 4, 5, 6]: Specific feature
            - 7: Newly introduced node
        - erg_edge_index (numpy.ndarray): Array of edges of ErG.
        - num_of_rings (int): Number of rings in the ErG.
    """

    # Array of atom features
    # 0: No feature, [1, 2, 3, 4, 5, 6]: Specific feature, 7: Newly introduced node
    atom_features = np.zeros(erg_fp.GetNumAtoms(), dtype=int)

    # Array of edges of ErG
    erg_edge_index = np.empty((2, 2 * erg_fp.GetNumBonds()), dtype=int)

    num_of_rings = 0
    edge_count = 0

    for i, atom in enumerate(erg_fp.GetAtoms()):
        # Assign features to atoms and edges and construct a graph
        if atom.GetSymbol() == "*":
            num_of_rings += 1

        for prop in atom.GetProp("_ErGAtomTypes"):
            if prop in ["0", "1", "2", "4", "5", "3"]:
                atom_features[i] = int(prop) + 1

        for bond in atom.GetBonds():
            # Construct edges of a graph
            erg_edge_index[0, edge_count] = i
            if i == bond.GetBeginAtomIdx():
                erg_edge_index[1, edge_count] = bond.GetEndAtomIdx()
            else:
                erg_edge_index[1, edge_count] = bond.GetBeginAtomIdx()

            edge_count += 1  # Each edge is counted twice

    return atom_features, erg_edge_index, num_of_rings


def createErgRingMapping(molecule, erg_num_atoms, num_of_rings):
    """
    Create ring mapping from the original graph to ErG.

    Parameters:
        - molecule (Chem.rdchem.Mol): RDKit Mol object representing a molecule.
        - erg_num_atoms (int): Number of atoms of ErG
        - num_of_rings (int): Number of rings in the molecule.

    Returns:
        - rings_map (numpy.ndarray): Array with two rows representing the mapping of nodes of the original graph to rings in ErG.
          The first row contains indices of atoms from the original graph mapped to a ring atoms of ErG.
          The second row contains indices of the corresponding ring nodes in the resulting ErG.
    """
    recognized_rings = findRecognizedRings(molecule)

    cumulative_ring_sizes = np.append(0, np.cumsum(list(map(len, recognized_rings))))

    rings_map = np.empty((2, int(cumulative_ring_sizes[-1])), dtype=int)
    rings_map_idx = 0

    for i, ring in enumerate(recognized_rings):
        idcs = np.arange(cumulative_ring_sizes[i], cumulative_ring_sizes[i + 1])
        rings_map[0, idcs] = ring

        # Nodes of ErG representing rings are inserted at the end of the list of nodes
        rings_map[1, idcs] = erg_num_atoms - num_of_rings + i

    return rings_map


def createErgMapping(molecule, num_of_rings, erg_num_atoms, atoms_with_properties_flattened):
    """
    Create a mapping for the Extended Graph (ErG) from a given molecule.

    Parameters:
        - molecule (Chem.rdchem.Mol): RDKit Mol object representing a molecule.
        - num_of_rings (int): Number of rings in the molecule.
        - erg_num_atoms (int): Number of atoms in the resulting ErG.
        - atoms_with_properties_flattened (list): List of atom indices with specific properties.

    Returns:
        - erg_mapping (numpy.ndarray): Array representing the mapping of atoms to ErG.
            - The first row contains indices of atoms from the original graph mapped to ErG atoms.
            - The second row contains indices of the corresponding ErG atoms.
        - unmapped (bool): Flag denoting if some atoms of raw graph are not mapped to any ErG node
    """

    # Create a mapping for atoms that are part of recognized rings
    rings_map = createErgRingMapping(molecule, erg_num_atoms, num_of_rings)

    # Mapping for atoms with specific properties
    prop_map = np.empty((2, molecule.GetNumAtoms()), dtype=int)
    prop_map_idx = 0

    # Mapping for unmapped atoms that are part of rings of length >= 8 and don't have specific properties
    unmapped = False
    unmapped_map = np.empty((2, molecule.GetNumAtoms()), dtype=int)
    unm_map_idx = 0

    for i, atom in enumerate(molecule.GetAtoms()):
        if not atom.IsInRing() or atom.GetDegree() != 2 or i in atoms_with_properties_flattened:
            # Atoms with specific properties
            prop_map[0, prop_map_idx] = i
            prop_map[1, prop_map_idx] = prop_map_idx
            prop_map_idx += 1
        elif molecule.GetRingInfo().MinAtomRingSize(i) >= 8:
            # Atoms without specific properties that are part of rings of length >= 8 are not mapped to any ErG atom.
            # All those atoms are mapped to and artifically introduced node of ErG with feature 7.
            unmapped_map[0, unm_map_idx] = i
            unmapped_map[1, unm_map_idx] = erg_num_atoms
            unm_map_idx += 1
            unmapped = True

    # Concatenate mappings
    erg_mapping = np.concatenate(
        (prop_map[:, :prop_map_idx], rings_map, unmapped_map[:, :unm_map_idx]), axis=1
    )

    return erg_mapping, unmapped


def getErGData(molecule, num_of_nodes):
    """
    Get data for the Extended Reduced Graph (ErG) from a given molecule.

    Parameters:
        - molecule (Chem.rdchem.Mol): RDKit Mol object representing a molecule.
        - num_of_nodes (int): Number of nodes in the raw graph.

    Returns:
        - data (ReducedGraphData): Data for the ErG in the form of ReducedGraphData, containing features and mapping.
    """

    # Generate ErG fingerprint
    erg_fp = GenerateMolExtendedReducedGraph(molecule)
    erg_num_atoms = erg_fp.GetNumAtoms()

    # Construct ErG features and edges
    atom_features, erg_edge_index, num_of_rings = constructErgFromFp(erg_fp)

    # Find atoms with specific properties in the molecule
    atoms_with_properties = moleculeAtomsProperties(molecule)
    atoms_with_properties_flattened = np.unique(
        np.array([index for tpl in atoms_with_properties for index in tpl]).flatten()
    )

    # Create mapping for ErG
    erg_mapping, unmapped = createErgMapping(
        molecule, num_of_rings, erg_num_atoms, atoms_with_properties_flattened
    )

    if unmapped:
        # If there are unmapped atoms, insert and artificall node
        erg_num_atoms += 1
        atom_features = np.append(atom_features, 7)

    # Transform ErG graph into ReducedGraphData
    data = ReducedGraphData()
    data.rg_edge_index = torch.from_numpy(erg_edge_index)
    data.mapping = torch.from_numpy(erg_mapping)
    data.rg_atom_features = torch.from_numpy(atom_features)
    data.rg_num_atoms = torch.tensor(erg_num_atoms, dtype=torch.int64)
    data.raw_num_atoms = num_of_nodes

    return data
