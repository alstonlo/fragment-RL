import itertools

from src.chem.fragment_utils import Fragment, combine
from src.chem.mol_utils import check_validity


class FragmentBasedDesigner:

    def __init__(self, init_mol, vocab, prop_fn, max_mol_size, max_steps, discount):
        self.init_mol = init_mol
        self.vocab = vocab
        self.prop_fn = prop_fn
        self.max_mol_size = max_mol_size
        self.max_steps = max_steps
        self.discount = discount

        self.state = (init_mol, self.max_steps)
        self.valid_actions = self._enum_valid_actions()

    @property
    def mol(self):
        return self.state[0]

    @property
    def steps_left(self):
        return self.state[1]

    @property
    def done(self):
        return not self.valid_actions

    def reset(self):
        self.state = (self.init_mol, self.max_steps)
        self.valid_actions = self._enum_valid_actions()

    def step(self, action):
        if self.done or (action not in self.valid_actions):
            raise ValueError

        # attach fragment vocab[action[1] onto action[0] of current state
        skeleton = Fragment(self.mol, action[0])
        arm = self.vocab[action[1]]
        new_mol = combine(skeleton=skeleton, arm=arm)
        assert check_validity(new_mol)

        self.state = (new_mol, self.steps_left - 1)
        self.valid_actions = self._enum_valid_actions()

        reward = self._reward_fn()
        if not self.valid_actions:
            while self.steps_left > 0:
                self.state = (new_mol, self.steps_left - 1)
                reward += self._reward_fn()

        return self.state, reward

    def _enum_valid_actions(self):
        if self.steps_left <= 0:
            return set()

        mol_size = self.mol.GetNumAtoms()
        valid_actions = set()
        for action in itertools.product(range(mol_size), range(len(self.vocab))):
            atom = self.mol.GetAtomWithIdx(action[0])
            arm = self.vocab[action[1]]

            if mol_size + arm.mol.GetNumAtoms() > self.max_mol_size:
                continue
            if atom.GetImplicitValence() == 0:
                continue
            valid_actions.add(action)
        return valid_actions

    def _reward_fn(self):
        return self.prop_fn(self.mol) * (self.discount ** self.steps_left)
