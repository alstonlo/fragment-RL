from rdkit import Chem


def check_validity(mol):
    if mol.GetNumBonds() < 1:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except ValueError:
        return False
