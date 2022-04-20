import pathlib
import statistics

from rdkit import Chem
import pandas as pd

from src.chem.mol_utils import Molecule, pairwise_diversities, uniqueness, validity


def main():
    result_dir = pathlib.Path(__file__).parents[2] / "results" / "qed_dqn"
    result_dir.mkdir(exist_ok=True)

    for epsilon in [0.0, 0.05, 0.1]:
        print(f"Epsilon={epsilon}")

        # Cut down number of samples to compare fairly with previous works
        sampled = pd.read_csv(result_dir / f"eps={epsilon}.csv")
        smiles = list(sampled["smiles"])[:100]
        values = list(sampled["value"])[:100]
        qeds = list(sampled["qed"])[:100]

        if epsilon == 0.0:
            smiles = smiles * 100

        print(f"\tQED:      ${statistics.mean(qeds):.3f}\\pm {statistics.pstdev(qeds):.3f}$")
        print(f"\tReturn:   ${statistics.mean(values):.3f}\\pm {statistics.pstdev(values):.3f}$")

        mols = [Molecule.from_smiles(s) for s in smiles]
        print(f"\tValid:    {validity(mols)}")
        print(f"\tUnique:   {uniqueness(mols):.3f}")
        print(f"\tDiverse:  {statistics.mean(pairwise_diversities(mols)):.3f}")

        top3 = list(sorted(qeds, reverse=True))[:3]
        if epsilon != 0:
            print(f"\tTop 3: [{top3[0]:.3f}, {top3[1]:.3f}, {top3[2]:.3f}]")

        print()


if __name__ == "__main__":
    main()
