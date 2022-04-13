import itertools

import torch

from src.chem.fragment_utils import Fragment, combine
from src.chem.mol_utils import sanitize, mol_to_dgl


class FragmentBasedDesigner:

    def __init__(self, init_mol, vocab, prop_fn, max_mol_size, max_steps, discount):
        self.init_mol = init_mol
        self.vocab = vocab
        self.prop_fn = prop_fn
        self.max_mol_size = max_mol_size
        self.max_steps = max_steps
        self.discount = discount
        self.nop_action = (0, len(vocab))

        self.state = (init_mol, self.max_steps)
        self.valid_actions = self._enum_valid_actions()

    @property
    def torch_state(self):
        time = self.steps_left / self.max_steps
        g = mol_to_dgl(self.mol, time)

        # turn valid actions into mask
        mask = torch.zeros((self.mol.GetNumAtoms(), len(self.vocab)))
        if self.valid_actions:
            a, b = tuple(zip(*list(self.valid_actions)))
            mask[a, b] = 1

        return g, mask

    @property
    def mol(self):
        return self.state[0]

    @property
    def steps_left(self):
        return self.state[1]

    @property
    def done(self):
        return self.steps_left == 0

    def reset(self):
        self.state = (self.init_mol, self.max_steps)
        self.valid_actions = self._enum_valid_actions()

    def step(self, action, commit=True):
        if self.done or (action not in self.valid_actions):
            raise ValueError

        if action == self.nop_action:
            new_mol = self.mol
        else:
            skeleton = Fragment(self.mol, action[0])
            arm = self.vocab[action[1]]
            new_mol = combine(skeleton=skeleton, arm=arm)
            assert sanitize(new_mol)

        reward = self.prop_fn(new_mol) * (self.discount ** (self.steps_left - 1))

        if commit:
            self.state = (new_mol, self.steps_left - 1)
            self.valid_actions = self._enum_valid_actions()
        return reward

    def _enum_valid_actions(self):
        if self.steps_left <= 0:
            return set()

        valid_actions = {self.nop_action}
        for action in itertools.product(range(self.mol.GetNumAtoms()), range(len(self.vocab))):
            atom = self.mol.GetAtomWithIdx(action[0])
            arm = self.vocab[action[1]]

            if self.mol.GetNumAtoms() + arm.mol.GetNumAtoms() > self.max_mol_size:
                continue
            if atom.GetImplicitValence() == 0:
                continue
            valid_actions.add(action)
        return valid_actions