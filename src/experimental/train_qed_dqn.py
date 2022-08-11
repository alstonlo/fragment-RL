import argparse
import pathlib

import torch
import wandb
from rdkit import Chem

from src.chem.prop_utils import qed
from src.data.environment import FragmentBasedDesigner
from src.data.vocab import FragmentVocab
from src.experimental.train_utils import train_dqn, seed_everything
from src.models.dqn import FragmentDQN

from src.experimental.docking_simple import DockingVina

PROJECT_DIR = pathlib.Path(__file__).parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--seed", type=int, default=413)

    parser.add_argument("--vocab_size", type=int, default=1000)

    parser.add_argument("--init_mol", type=str, default="CC")
    parser.add_argument("--max_mol_size", type=int, default=38)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--discount", type=float, default=0.7)

    parser.add_argument("--n_node_hidden", type=int, default=128)
    parser.add_argument("--n_edge_hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)

    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--n_train_iters", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--update_freq", type=int, default=1000)
    parser.add_argument("--polyak", type=float, default=0.0)
    parser.add_argument("--eps_decay", type=float, default=0.9999)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--ckpt_freq", type=int, default=2000)

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = FragmentVocab.load_from_pkl(PROJECT_DIR / "data" / "vocab.pkl")
    vocab.cull(args.vocab_size)

    docking_config = {'vina_program': 'bin/qvina2.exe', 'temp_dir': 'tmp', 'exhaustiveness': 1,
                      'num_sub_proc': 10, 'num_cpu_dock': 5, 'num_modes': 10, 'timeout_gen3d': 30,
                      'timeout_dock': 100, 'receptor_file': 'docking/fa7/receptor.pdbqt'}
    box_center = (26.413, 11.282, 27.238)
    box_size = (18.521, 17.479, 19.995)
    docking_config['box_parameter'] = (box_center, box_size)
    dv = DockingVina(docking_config)

    env = FragmentBasedDesigner(
        init_mol=Chem.MolFromSmiles(args.init_mol),
        vocab=vocab,
        prop_fn=dv.predict,
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
        wandb.save("*.pt")

    seed_everything(args.seed)
    train_dqn(dqn=dqn, env=env, **vars(args))


if __name__ == "__main__":
    main()
