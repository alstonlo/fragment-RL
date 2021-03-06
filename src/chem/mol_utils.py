import dgl
import torch
import functools
import itertools
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import HybridizationType, BondType
from chem.prop_utils import qed

class Molecule:

    @classmethod
    def from_smiles(cls, smiles):
        return Molecule(Chem.MolFromSmiles(smiles))

    def __init__(self, rdkmol):
        self.smiles = Chem.MolToSmiles(rdkmol)

    def __str__(self):
        return self.base_smiles

    def __hash__(self):
        return hash(self.smiles)

    def __eq__(self, other):
        return self.smiles == other.smiles

    @property
    @functools.lru_cache()
    def rdkmol(self):
        return Chem.MolFromSmiles(self.smiles)

    @property
    @functools.lru_cache()
    def base_smiles(self):
        clone = Chem.Mol(self.rdkmol)
        for atom in clone.GetAtoms():
            if openness(atom):
                set_openness(atom, is_open=False)
        return Chem.MolToSmiles(clone)

    def base_copy(self):
        return Molecule.from_smiles(self.base_smiles)

    def visualize(self, width=300, height=300):
        rdDepictor.Compute2DCoords(self.rdkmol)
        rdDepictor.GenerateDepictionMatching2DStructure(self.rdkmol, self.rdkmol)
        return Draw.MolToImage(self.rdkmol, size=(width, height))

def draw_mol_to_file(mol_lst, directory):
    mols = []
    for molecule in mol_lst:
        smiles = molecule.smiles
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)

    img = Draw.MolsToGridImage(mols,molsPerRow=3,subImgSize=(200,200),legends=[f"QED = {round(qed(x), 3)}" for x in mols]) 
    img.save(directory+'/'+smiles+'.png')    
    
def sanitize(mol):
    if mol.GetNumBonds() < 1:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except ValueError:
        return False


# =============================================================================
# DGL Utils
# =============================================================================

ATOM_TYPES = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
HYBRID_TYPES = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3]
BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, None]

def openness(atom):
    return atom.GetAtomMapNum() == 1

def _zinc_atoms_features(mol, time):
    feats = {"node_type": [], "node_charge": [], "n_feat": []}

    for u in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(u)
        charge = atom.GetFormalCharge()
        symbol = atom.GetSymbol()
        atom_type = atom.GetAtomicNum()
        aromatic = atom.GetIsAromatic()
        hybridization = atom.GetHybridization()
        num_h = atom.GetTotalNumHs()

        feats["node_type"].append(atom_type)
        feats["node_charge"].append(charge)

        h_u = []
        h_u += [int(symbol == x) for x in ATOM_TYPES]
        h_u.append(atom_type)
        h_u.append(int(charge))
        h_u.append(int(aromatic))
        h_u += [int(hybridization == x) for x in HYBRID_TYPES]
        h_u.append(num_h)
        h_u.append(time)
        feats["n_feat"].append(torch.tensor(h_u, dtype=torch.float))

    feats["n_feat"] = torch.stack(feats["n_feat"], dim=0)
    feats["node_type"] = torch.tensor(feats["node_type"], dtype=torch.long)
    feats["node_charge"] = torch.tensor(feats["node_charge"], dtype=torch.long)
    return feats


def _zinc_edge_features(mol, time, edges, self_loop=False):
    feats = {"e_feat": []}

    edges = [idxs.tolist() for idxs in edges]
    for e in range(len(edges[0])):
        u, v = edges[0][e], edges[1][e]
        if u == v and not self_loop:
            continue

        e_uv = mol.GetBondBetweenAtoms(u, v)
        if e_uv is None:
            bond_type = None
        else:
            bond_type = e_uv.GetBondType()

        e_uv = []
        e_uv += [float(bond_type == x) for x in BOND_TYPES]
        e_uv.append(time)
        feats["e_feat"].append(e_uv)

    feats["e_feat"] = torch.tensor(feats["e_feat"], dtype=torch.float)
    return feats


def mol_to_dgl(mol, time):
    g = dgl.graph([])

    num_atoms = mol.GetNumAtoms()
    atom_feats = _zinc_atoms_features(mol, time)
    g.add_nodes(num=num_atoms, data=atom_feats)

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        g.add_edges([u, v], [v, u])

    bond_feats = _zinc_edge_features(mol, time, g.edges())
    g.edata.update(bond_feats)
    return g

# ==================================================================================================
# Metrics
# ==================================================================================================


def validity(mols):  # validity guaranteed, but sanity check
    valid = [m for m in mols if (Chem.MolFromSmiles(m.base_smiles) is not None)]
    return len(valid) / len(mols)


def uniqueness(mols):
    base_smiles = [m.base_smiles for m in mols]
    return len(set(base_smiles)) / len(base_smiles)


def molecule_similarity(mol1, mol2):
    vec1 = rdMolDescriptors.GetMorganFingerprint(mol1.rdkmol, radius=2)
    vec2 = rdMolDescriptors.GetMorganFingerprint(mol2.rdkmol, radius=2)
    return DataStructs.TanimotoSimilarity(vec1, vec2)


def pairwise_diversities(mols):
    scores = []
    for mol1, mol2 in itertools.combinations(mols, r=2):
        scores.append(1 - molecule_similarity(mol1, mol2))
    return scores

