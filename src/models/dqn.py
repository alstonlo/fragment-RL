import torch
import torch.nn as nn

from src.models.modules import MLP, GraphEncoder


class FragmentDQN(nn.Module):

    def __init__(self, n_node_hidden, n_edge_hidden, n_layers, vocab_size, **kwargs):
        super().__init__()

        self.gnn = GraphEncoder(
            n_atom_feat=18,  # hardcoded
            n_bond_feat=6,  # hardcoded
            n_node_hidden=n_node_hidden,
            n_edge_hidden=n_edge_hidden,
            n_layers=n_layers
        )

        self.mlp = MLP(n_node_hidden, vocab_size)

    def forward(self, state):
        g = state
        baseline = g.ndata["baseline"]

        h = self.gnn(g, g.ndata['n_feat'], g.edata['e_feat'])
        values = self.mlp(h)
        values = values + baseline
        values = torch.concat([values, baseline], dim=1)
        mask = torch.where(g.ndata["mask"], 0.0, float("-inf"))
        return values + mask
