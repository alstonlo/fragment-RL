import argparse
import pathlib
import torch
from rdkit import Chem

from src.chem.prop_utils import qed
from src.data.environment import FragmentBasedDesigner
from src.data.vocab import FragmentVocab
from src.models.dqn import DummyFragmentDQN
from src.models.replay_buffer import ReplayBuffer

PROJECT_DIR = pathlib.Path(__file__).parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=3)

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

    dqn = DummyFragmentDQN(n_feats=18, n_vocab=args.vocab_size)

    # buffer = ReplayBuffer(100)
    # for i in range(100):
    #     s_t = env.torch_state
    #     output = dqn(s_t)
    #     act = torch.argmax(output, dim=1)
    #     rew = env.step(act)
    #     s_tp1 = env.torch_state
    #     done = env.done
    #     buffer.add(s_t=s_t, act=act, rew=rew, s_tp1=s_tp1, done=done)
    #     if done:
    #         break
    #
    # s_ts, acts, rews, s_tp1s, dones = buffer.sample(100)

    # TODO: replace with actual DQN
    dqn = DummyFragmentDQN(n_feats=18, n_vocab=args.vocab_size)
    g, m = env.torch_state
    print(g.ndata["n_feat"])
    graphwithmask = dqn(env.torch_state)
    print(graphwithmask)
if __name__ == "__main__":
    main()
