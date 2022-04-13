import argparse
import pathlib

from rdkit import Chem

from src.chem.prop_utils import qed
from src.data.environment import FragmentBasedDesigner
from src.data.vocab import FragmentVocab

PROJECT_DIR = pathlib.Path(__file__).parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=1000)

    parser.add_argument("--init_mol", type=str, default="CC")
    parser.add_argument("--max_mol_size", type=int, default=38)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--discount", type=float, default=0.9)
    args = parser.parse_args()

    vocab = FragmentVocab.load_from_pkl(PROJECT_DIR / "data" / "vocab.pkl")
    vocab.cull(args.vocab_size)

    env = FragmentBasedDesigner(
        init_mol=Chem.MolFromSmiles("CC"),
        vocab=vocab,
        prop_fn=qed,
        max_mol_size=args.max_mol_size,
        max_steps=args.max_steps,
        discount=args.discount
    )

    env.reset()


if __name__ == "__main__":
    main()
