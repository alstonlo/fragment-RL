import argparse
import pathlib

from src.data.vocab import FragmentVocab

PROJECT_DIR = pathlib.Path(__file__).parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=1000)
    args = parser.parse_args()

    vocab = FragmentVocab.load_from_pkl(PROJECT_DIR / "data" / "vocab.pkl")
    vocab.cull(args.vocab_size)


if __name__ == "__main__":
    main()
