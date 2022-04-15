import random

import dgl
import torch


# Reference: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ReplayBuffer:

    def __init__(self, size):
        self.storage = []
        self.capacity = size
        self.next_idx = 0

    def __len__(self):
        return len(self.storage)

    def add(self, s_t, act, rew, s_tp1, done):
        data = (s_t, act, rew, s_tp1, done)

        if self.next_idx >= len(self):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size):
        k = min(batch_size, len(self))
        batch = random.sample(self.storage, k=k)
        s_ts, acts, rews, s_tp1s, dones = tuple(zip(*batch))

        s_ts = dgl.batch(s_ts)
        acts = torch.tensor(acts, dtype=torch.long)
        rews = torch.tensor(rews, dtype=torch.float)
        s_tp1s = dgl.batch(s_tp1s)
        dones = torch.tensor(dones, dtype=torch.long)

        return s_ts, acts, rews, s_tp1s, dones
