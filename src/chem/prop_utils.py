from rdkit.Chem import QED


def qed(mol):
    try:
        return QED.qed(mol)
    except ValueError:
        return 0.0
