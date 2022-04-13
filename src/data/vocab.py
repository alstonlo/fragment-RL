import pickle

from rdkit.Chem.rdchem import BondType

from src.chem.fragments import break_single_bond


class FragmentVocab:

    @classmethod
    def load_from_pkl(cls, file_path):
        with open(file_path, "rb") as cache:
            return pickle.load(cache)

    @classmethod
    def extract_from_mols(cls, mols, arm_size=10):
        log = {}

        def _keep_arm(_skeleton, _arm):
            if _skeleton.mol.GetAtomWithIdx(_skeleton.root).GetAtomicNum() != 6:
                return False
            if _arm.mol.GetNumAtoms() > arm_size:
                return False
            if not any(a.GetAtomicNum() != 6 for a in _arm.mol.GetAtoms()):
                return False
            if not any(b.GetBondType() in {BondType.DOUBLE, BondType.TRIPLE} for b in _arm.mol.GetBonds()):
                return False
            if _arm.smiles.startswith("CC"):
                return False
            return True

        for mol in mols:
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()

                try:
                    f1, f2 = break_single_bond(mol, u, v)
                except ValueError:
                    continue

                for skeleton, arm in [(f1, f2), (f2, f1)]:
                    if _keep_arm(_skeleton=skeleton, _arm=arm):
                        log.setdefault(arm.smiles, arm).count += 1

        arms = list(log.values())
        return FragmentVocab(arms)

    def __init__(self, arms):
        self.arms = list(arms)
        self.arms.sort(key=lambda f: f.count, reverse=True)

    def __len__(self):
        return len(self.arms)

    def cull(self, vocab_size):
        self.arms = self.arms[:vocab_size]
