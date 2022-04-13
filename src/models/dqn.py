import torch
import torch.nn as nn


class DummyFragmentDQN:
    # for debugging, just applies MLP on node features

    def __init__(self, n_feats, n_vocab):
        self.n_vocab = n_vocab

        self.lin = nn.Sequential(
            nn.Linear(n_feats, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_vocab)
        )

    def forward(self, state):
        graphs, masks = state
        values = self.lin(graphs.ndata["n_feat"])
        mask = torch.where(masks, 0, float("-inf"))
        return values + mask
