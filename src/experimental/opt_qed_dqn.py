import pathlib

import pandas as pd
import torch
from tqdm import trange
from rdkit import Chem

from src.chem.prop_utils import qed
from src.models.agents import DQNAgent
from src.models.dqn import FragmentDQN
from src.data.environment import FragmentBasedDesigner
from train_utils import seed_everything
from src.data.vocab import FragmentVocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = pathlib.Path(__file__).parents[2]

def main():
    result_dir = pathlib.Path(__file__).parents[2] / "results" / "qed_dqn"
    result_dir.mkdir(exist_ok=True)

    dqn = torch.load(result_dir / "model.pt", map_location=DEVICE)
    dqn.device = DEVICE

    env = FragmentBasedDesigner(
        init_mol=Chem.MolFromSmiles("CC"),
        vocab=FragmentVocab.load_from_pkl(PROJECT_DIR / "data" / "vocab.pkl"),
        prop_fn=qed,
        max_mol_size=38,
        max_steps=15,
        discount=0.7
    )

    for epsilon in [0.0, 0.05, 0.1]:
        seed_everything(seed=498)
        agent = DQNAgent(dqn, epsilon, DEVICE)

        sampled = []
        n_samples = 1 if epsilon == 0.0 else 150
        for _ in trange(n_samples, desc=f"Eps={epsilon:.2f}"):
            mol, value = agent.rollout(env)
            next_qed = env.prop_fn(mol)
            sampled.append({"smiles": str(mol), "value": value, "qed": next_qed})
        sampled = pd.DataFrame(sampled)
        sampled.to_csv(result_dir / f"eps={epsilon}.csv", index=False)


if __name__ == "__main__":
    main()
