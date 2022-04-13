import random

import torch


class Agent:

    def sample_action(self, env):
        raise NotImplementedError()

    def rollout(self, env):
        env.reset()

        value = 0.0
        while not env.done:
            act = self.sample_action(env)
            rew = env.step(act)
            value += rew
        return env.mol, value


class EpsilonGreedyAgent(Agent):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def sample_action(self, env):
        if random.random() < self.epsilon:
            return random.choice(env.valid_actions)
        else:
            best = (None, float("-inf"))
            for action in env.valid_actions:
                score = env.prop_fn(action)
                if score > best[1]:
                    best = (action, score)
            return best[0]


class DQNAgent(Agent):

    def __init__(self, dqn, epsilon):
        self.dqn = dqn
        self.epsilon = epsilon

    def sample_action(self, env):
        action_space = env.valid_actions

        if random.random() < self.epsilon:
            return random.choice(action_space)
        else:
            future_obses = [(a, env.state[1] - 1) for a in action_space]
            pred_values = self.dqn(future_obses).squeeze(1)
            best = torch.argmax(pred_values).item()
            return action_space[best]