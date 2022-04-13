import copy
import pathlib
import statistics

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from src.agents import DQNAgent
from src.utils.replay_buffer import ReplayBuffer
from tqdm import trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wandb_checkpoint(model):
    model_path = str(pathlib.Path(wandb.run.dir) / "model.pt")
    torch.save(model, model_path)
    wandb.save(model_path)


def target_dqn_update(dqn, target_dqn, polyak):
    with torch.no_grad():
        for p, target_p in zip(dqn.parameters(), target_dqn.parameters()):
            target_p.data.mul_(polyak)
            target_p.data.add_((1 - polyak) * p.data)


def dqn_update(dqn, target_dqn, batch, optimizer, scheduler):
    sa_ts, rewards, sa_tp1ses, dones = batch
    sa_ts = np.concatenate(sa_ts, axis=0)
    rewards = torch.tensor(rewards, dtype=torch.float, device=DEVICE)

    with torch.no_grad():
        opt_actions = torch.zeros(sa_ts.shape, device=DEVICE)
        for i, sa_tp1s in enumerate(sa_tp1ses):
            if not dones[i]:
                sa_tp1s = torch.tensor(sa_tp1s, dtype=torch.float, device=DEVICE)
                opt_index = torch.argmax(dqn(sa_tp1s)).item()
                opt_actions[i] = sa_tp1s[opt_index]

        mask = 1 - torch.tensor(dones, dtype=torch.float, device=DEVICE)
        v_tp1s = mask * target_dqn(opt_actions).squeeze(1)

    dqn.train()
    td_target = rewards + v_tp1s
    q_ts = dqn(sa_ts).squeeze(1)
    loss = F.huber_loss(td_target, q_ts)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dqn.parameters(), 10)
    optimizer.step()
    scheduler.step()

    dqn.eval()
    return loss.item()


def train_double_dqn(
        dqn, env, buffer_size,
        n_episodes, batch_size, lr,
        learn_freq, update_freq, polyak,
):
    assert env.max_steps % learn_freq == 0
    assert env.max_steps % update_freq == 0
    dqn.eval()

    behavior_agent = DQNAgent(dqn, epsilon=1.0)
    optimal_agent = DQNAgent(dqn, epsilon=0.0)
    eps_step = 0.99 / n_episodes

    replay_buffer = ReplayBuffer(buffer_size)
    target_dqn = copy.deepcopy(dqn).to(DEVICE)
    for p in target_dqn.parameters():
        p.requires_grad = False
    target_dqn.eval()

    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)

    for episode in trange(n_episodes, desc="Episodes"):
        env.reset()
        losses = []

        for step in range(env.max_steps):

            with torch.no_grad():
                act = behavior_agent.sample_action(env)
            next_obs, reward, done = env.step(act)

            # since MDP is deterministic (s, a) can be represented with s'
            sa_t = dqn.featurize_batch([next_obs])
            if done:
                sa_tp1s = None
            else:
                sa_tp1s = [(a, env.state[1] - 1) for a in env.valid_actions]
                sa_tp1s = dqn.featurize_batch(sa_tp1s)
            replay_buffer.add(sa_t, reward, sa_tp1s, done)

            # perform double DQN update
            if ((step + 1) % learn_freq == 0) and (len(replay_buffer) > 1):
                batch = replay_buffer.sample(batch_size)
                loss = dqn_update(dqn, target_dqn, batch, optimizer, scheduler)
                losses.append(loss)
            if (step + 1) % update_freq == 0:
                target_dqn_update(dqn, target_dqn, polyak)

        behavior_agent.epsilon -= eps_step
        avg_loss = statistics.mean(losses)

        # try rolling out optimally
        env.reset()
        mol, value = optimal_agent.rollout(env)
        qed = env.prop_fn(mol)

        # wandb logging
        metrics = {"Episode": episode, "Value": value, "QED": qed, "Loss": avg_loss}
        if episode % 20 == 0:
            metrics["Molecule"] = wandb.Image(mol.visualize())
        wandb.log(metrics)

        if episode % 100 == 0:
            wandb_checkpoint(model=dqn)
