from rdkit.Chem import QED

from src.utils.mol_utils import enum_molecule_mods, Molecule


class ScaffoldDecorator:

    def __init__(
            self,
            init_mol,
            prop_fn,
            atom_types,
            allowed_ring_sizes,
            max_mol_size,
            max_steps,
            discount
    ):
        self.init_mol = init_mol
        self.prop_fn = prop_fn
        self.atom_types = atom_types
        self.allowed_ring_sizes = allowed_ring_sizes
        self.max_mol_size = max_mol_size
        self.max_steps = max_steps
        self.discount = discount

        self._mol = None
        self._steps_left = self.max_steps
        self._valid_actions = set()

    @property
    def state(self):
        return self._mol, self._steps_left

    @property
    def valid_actions(self):
        if self._steps_left == 0:
            return list()
        return list(sorted(self._valid_actions, key=lambda m: m.smiles))

    def reset(self):
        self._mol = self.init_mol
        self._steps_left = self.max_steps
        self._rebuild_valid_actions()

    def step(self, action):
        # Assumes action (next molecule) is valid
        # Returns (next state, reward, done)
        assert isinstance(action, Molecule)

        if self._steps_left == 0:
            raise ValueError()
        elif action not in self._valid_actions:
            return ValueError()
        changed = (action != self._mol)

        self._mol = action
        self._steps_left -= 1
        if changed:
            self._rebuild_valid_actions()
        if self._steps_left == 1:  # set all actions to terminal
            self._valid_actions = set(m.base_copy() for m in self._valid_actions)

        reward = self._reward_fn()
        done = (self._steps_left == 0)
        return self.state, reward, done

    def _rebuild_valid_actions(self):
        self._valid_actions = enum_molecule_mods(
            mol=self._mol,
            atom_types=self.atom_types,
            allowed_ring_sizes=self.allowed_ring_sizes,
            max_mol_size=self.max_mol_size
        )

    def _reward_fn(self):
        return self.prop_fn(self._mol) * (self.discount ** self._steps_left)


class QEDScaffoldDecorator(ScaffoldDecorator):

    def __init__(
            self,
            init_mol,
            atom_types=("C", "O", "N"),
            allowed_ring_sizes=(5, 6),
            max_mol_size=38,
            max_steps=40,
            discount=0.9
    ):
        super().__init__(
            init_mol=init_mol,
            prop_fn=self.qed,
            atom_types=atom_types,
            allowed_ring_sizes=allowed_ring_sizes,
            max_mol_size=max_mol_size,
            max_steps=max_steps,
            discount=discount
        )

    def qed(self, mol):
        try:
            return QED.qed(mol.rdkmol)
        except ValueError:
            return 0.0
