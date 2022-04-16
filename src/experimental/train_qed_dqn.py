import argparse
import pathlib

import torch
import wandb
from rdkit import Chem

from src.chem.prop_utils import qed
from src.data.environment import FragmentBasedDesigner
from src.data.vocab import FragmentVocab
from src.experimental.train_utils import train_double_dqn, seed_everything
from src.models.dqn import FragmentDQN

PROJECT_DIR = pathlib.Path(__file__).parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--seed", type=int, default=413)

    parser.add_argument("--vocab_size", type=int, default=1000)

    parser.add_argument("--init_mol", type=str, default="CC")
    parser.add_argument("--max_mol_size", type=int, default=38)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--discount", type=float, default=0.7)

    parser.add_argument("--n_node_hidden", type=int, default=64)
    parser.add_argument("--n_edge_hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)

    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--n_train_iters", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--update_freq", type=int, default=20)
    parser.add_argument("--polyak", type=float, default=0.0)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--ckpt_freq", type=int, default=1000)

    args = parser.parse_args()
    args.device = "gpu" if torch.cuda.is_available() else "cpu"

    vocab = FragmentVocab.load_from_pkl(PROJECT_DIR / "data" / "vocab.pkl")
    vocab.cull(args.vocab_size)

    env = FragmentBasedDesigner(
        init_mol=Chem.MolFromSmiles(args.init_mol),
        vocab=vocab,
        prop_fn=qed,
        max_mol_size=args.max_mol_size,
        max_steps=args.max_steps,
        discount=args.discount
    )

    seed_everything(args.seed)
    dqn = FragmentDQN(**vars(args))

    args.n_params = sum(p.numel() for p in dqn.parameters())
    args.n_params_trainable = sum(p.numel() for p in dqn.parameters() if p.requires_grad)

    # logging
    log_dir = pathlib.Path(__file__).parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        wandb.init(project="train_QED_FragmentDQN", dir=str(log_dir))
        wandb.config.update(vars(args))

    seed_everything(args.seed)
    train_double_dqn(dqn=dqn, env=env, **vars(args))


if __name__ == "__main__":
    main()
