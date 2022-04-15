import copy

import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import NNConv

class DummyFragmentDQN(nn.Module):
    # for debugging, just applies MLP on node features

    def __init__(self, n_feats, n_vocab):
        super().__init__()

        # self.loss_fn = torch.nn.HuberLoss()

        # self.GCN = nn.Sequential(
        #     NNConv(n_feats, n_feats)
        # )
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
        mask = torch.where(masks, 0.0, float("-inf"))
        return values + mask
