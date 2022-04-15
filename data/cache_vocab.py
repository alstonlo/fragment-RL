import pathlib
import pickle

from rdkit import Chem
from tqdm import tqdm

from src.data.vocab import FragmentVocab

if __name__ == "__main__":
    # script to cache extracted vocab from ChEMBL
    debug = False  # TODO: set to False

    def mol_iter(path):
        with open(path, "r") as f:
            lines = f.readlines()
            lines = lines[:10000] if debug else lines
        for line in tqdm(lines, desc="Extracting fragments"):
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            yield mol

    data_dir = pathlib.Path(__file__).parent
    chembl_path = data_dir / "chembl.txt"
    chembl_iter = mol_iter(chembl_path)

    vocab = FragmentVocab.extract_from_mols(chembl_iter)

    with open(data_dir / "vocab.pkl", "wb") as cache:
        pickle.dump(vocab, cache)
