import copy
import logging
import os
import pathlib
import random
import statistics

import dgl
import numpy as np
import torch
import torch.nn.functional  as F
import wandb
from rdkit.Chem import rdDepictor, Draw
from tqdm import trange

from src.models.agents import DQNAgent
from src.models.replay_buffer import ReplayBuffer

log = logging.getLogger(__name__)


def seed_everything(seed):
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def train_double_dqn(
        dqn, env, buffer_size,
        n_train_iters, batch_size, lr, update_freq, polyak,
        use_wandb, log_freq, device, **kwargs
):
    device = torch.device(device)
    dqn = dqn.to(device)
    dqn.eval()
    policy = DQNAgent(dqn, epsilon=1.0, device=device)

    target_dqn = copy.deepcopy(dqn).to(device)
    for p in target_dqn.parameters():
        p.requires_grad = False
    target_dqn.eval()

    buffer = ReplayBuffer(buffer_size)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)

    env.reset()
    episode = 0
    burn_in = batch_size

    for step in trange(n_train_iters, desc="Iteration"):

        # take step in environment
        s_t = env.torch_state
        act = policy.sample_action(env)
        rew = env.step(act)
        s_tp1 = env.torch_state
        done = env.done
        buffer.add(s_t=s_t, act=act, rew=rew, s_tp1=s_tp1, done=done)

        # DQN update step
        if step > burn_in:
            batch = buffer.sample(batch_size)
            loss = dqn_update(dqn, target_dqn, batch, optimizer, device)
            if use_wandb and (step % log_freq == 0):
                wandb.log({"Iteration": step, "Bellman Loss": loss})

        # target DQN soft-polyak update
        if (step + 1) % update_freq == 0:
            soft_target_update(dqn, target_dqn, polyak)

        if done:
            env.reset()
            episode += 1

            # validate and logging
            val_step(episode, dqn, env, use_wandb, device)

            # decay epsilon
            policy.epsilon = min(policy.epsilon * 0.999, 0.01)


def dqn_update(dqn, target_dqn, batch, optimizer, device):
    dqn.train()

    # unpack batch and move to devce
    batch = tuple(x.to(device) for x in batch)
    s_ts, acts, rews, s_tp1s, dones = batch

    # compute targets
    with torch.no_grad():
        s_tp1s.ndata["Q"] = target_dqn(s_tp1s)

        td_target = torch.clone(rews)
        for i, g in enumerate(dgl.unbatch(s_tp1s)):
            if not dones[i]:
                td_target[i] += g.ndata["Q"].max()

    offsets = torch.cumsum(s_ts.batch_num_nodes(), dim=0)
    offsets = torch.roll(offsets, shifts=1)
    offsets[0] = 0

    values = dqn(s_ts)
    Q_ts = values[acts[:, 0] + offsets, acts[:, 1]]

    loss = F.huber_loss(Q_ts, td_target)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dqn.parameters(), 10)
    optimizer.step()

    dqn.eval()
    return loss.item()


def soft_target_update(dqn, target_dqn, polyak):
    with torch.no_grad():
        for p, target_p in zip(dqn.parameters(), target_dqn.parameters()):
            target_p.data.mul_(polyak)
            target_p.data.add_((1 - polyak) * p.data)


def val_step(episode, dqn, env, use_wandb, device):
    eps_greedy = DQNAgent(dqn, epsilon=0.05, device=device)  # almost greedy
    greedy = DQNAgent(dqn, epsilon=0.0, device=device)

    eps_outs = []
    with torch.no_grad():
        for _ in range(10):
            eps_outs.append(eps_greedy.rollout(env))
        opt_mol, opt_value = greedy.rollout(env)
    eps_mols, eps_values = tuple(zip(*eps_outs))
    eps_QEDs = [env.prop_fn(mol) for mol in eps_mols]

    metrics = {
        "Episode": episode,
        "Value-e=0.05": statistics.fmean(eps_values),
        "Value-greedy": opt_value,
        "QED-e=0.05": statistics.fmean(eps_QEDs),
        "QED-greedy": env.prop_fn(opt_mol)
    }

    # wandb logging
    if use_wandb:
        if episode % 20 == 0:
            metrics["Mol-greedy"] = wandb.Image(visualize_mol(opt_mol))
            metrics["Mol-e=0.05"] = wandb.Image(visualize_mol(eps_mols[-1]))
        wandb.log(metrics)

        if episode % 100 == 0:
            model_path = str(pathlib.Path(wandb.run.dir) / "model.pt")
            torch.save(dqn, model_path)
            wandb.save(model_path)


def visualize_mol(mol, width=300, height=300):
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.GenerateDepictionMatching2DStructure(mol, mol)
    return Draw.MolToImage(mol, size=(width, height))
