import copy

from rdkit import Chem
from rdkit.Chem.rdchem import BondType

class Fragment:

    def __init__(self, mol, root):
        self.mol = mol
        self.root = root  # jagged open-end of fragment
        self.count = 0  # for vocab statistics

    @property
    def smiles(self):
        return Chem.MolToSmiles(self.mol, rootedAtAtom=self.root)


def break_single_bond(mol, u, v):
    mol = Chem.RWMol(copy.deepcopy(mol))
    bond = mol.GetBondBetweenAtoms(u, v)

    if bond.GetBondType() == BondType.SINGLE:
        mol.RemoveBond(u, v)
    else:
        raise ValueError

    mapping = []
    frags = list(Chem.rdmolops.GetMolFrags(mol, asMols=True, fragsMolAtomMapping=mapping))
    mapping = [list(m) for m in mapping]
    if not len(frags) == 2:
        raise ValueError

    if u not in mapping[0]:
        mapping, frags = mapping[::-1], frags[::-1]
    u = mapping[0].index(u)
    v = mapping[1].index(v)
    return Fragment(frags[0], u), Fragment(frags[1], v)


def combine(skeleton, arm):
    mol = Chem.CombineMols(skeleton.mol, arm.mol)
    mol = Chem.RWMol(mol)
    u = skeleton.root
    v = skeleton.mol.GetNumAtoms() + arm.root
    mol.AddBond(u, v, BondType.SINGLE)
    return mol.GetMol()
