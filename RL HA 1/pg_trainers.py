import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import PolicyNetwork, ValueNetwork
from rollout import collect_batch


def train_vanilla_pg(env, num_episodes=1000, batch_size=10, lr=1e-3,
                     gamma=0.99, device=None, seed=42):
    torch.manual_seed(seed)
    policy = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_history = []
    avg_reward_history = []

    for episode in range(0, num_episodes, batch_size):
        trajectories = collect_batch(env, policy, device, batch_size, gamma)

        policy_loss = 0
        for traj in trajectories:
            returns_tensor = torch.FloatTensor(traj['returns']).to(device)
            log_probs = traj['log_probs']
            if log_probs.dim() == 0:
                log_probs = log_probs.unsqueeze(0)
            policy_loss += -(log_probs * returns_tensor).sum()

        policy_loss /= batch_size

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        batch_rewards = [t['total_reward'] for t in trajectories]
        mean_reward = np.mean(batch_rewards)
        reward_history.extend(batch_rewards)
        avg_reward_history.append(mean_reward)

        if (episode // batch_size) % 20 == 0:
            print(f"[VanillaPG] Episode {episode}, Mean Reward: {mean_reward:.1f}")

    return policy, reward_history, avg_reward_history


def train_pg_average_baseline(env, num_episodes=1000, batch_size=10, lr=1e-3,
                               gamma=0.99, device=None, seed=42):
    """
    PG с baseline = скользящее среднее наград.
    Loss = -sum(log_prob * (G_t - baseline))
    """
    torch.manual_seed(seed)
    policy = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_history = []
    avg_reward_history = []
    running_baseline = 0.0 
    total_episodes_seen = 0

    for episode in range(0, num_episodes, batch_size):
        trajectories = collect_batch(env, policy, device, batch_size, gamma)

        batch_rewards = [t['total_reward'] for t in trajectories]

        policy_loss = 0
        for traj in trajectories:
            returns_tensor = torch.FloatTensor(traj['returns']).to(device)
            log_probs = traj['log_probs']
            if log_probs.dim() == 0:
                log_probs = log_probs.unsqueeze(0)
            advantage = returns_tensor - running_baseline
            policy_loss += -(log_probs * advantage).sum()

        policy_loss /= batch_size

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        for r in batch_rewards:
            total_episodes_seen += 1
            running_baseline += (r - running_baseline) / total_episodes_seen

        mean_reward = np.mean(batch_rewards)
        reward_history.extend(batch_rewards)
        avg_reward_history.append(mean_reward)

        if (episode // batch_size) % 20 == 0:
            print(f"[AvgBaseline] Episode {episode}, Mean Reward: {mean_reward:.1f}, "
                  f"Baseline: {running_baseline:.1f}")

    return policy, reward_history, avg_reward_history


def train_pg_value_baseline(env, num_episodes=1000, batch_size=10,
                             lr_policy=1e-3, lr_value=1e-3,
                             gamma=0.99, device=None, seed=42):

    torch.manual_seed(seed)
    policy = PolicyNetwork().to(device)
    value_net = ValueNetwork().to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr_policy)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)

    reward_history = []
    avg_reward_history = []

    for episode in range(0, num_episodes, batch_size):
        trajectories = collect_batch(env, policy, device, batch_size, gamma)

        policy_loss = 0
        value_loss = 0

        for traj in trajectories:
            states_tensor = torch.FloatTensor(traj['states']).to(device)
            returns_tensor = torch.FloatTensor(traj['returns']).to(device)
            log_probs = traj['log_probs']
            if log_probs.dim() == 0:
                log_probs = log_probs.unsqueeze(0)

            values = value_net(states_tensor)

            advantage = (returns_tensor - values.detach())

            policy_loss += -(log_probs * advantage).sum()
            value_loss += nn.functional.mse_loss(values, returns_tensor)

        policy_loss /= batch_size
        value_loss /= batch_size

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        batch_rewards = [t['total_reward'] for t in trajectories]
        mean_reward = np.mean(batch_rewards)
        reward_history.extend(batch_rewards)
        avg_reward_history.append(mean_reward)

        if (episode // batch_size) % 20 == 0:
            print(f"[ValueBaseline] Episode {episode}, Mean Reward: {mean_reward:.1f}")

    return policy, reward_history, avg_reward_history


def train_pg_rloo_baseline(env, num_episodes=1000, batch_size=10, lr=1e-3,
                            gamma=0.99, device=None, seed=42):

    torch.manual_seed(seed)
    policy = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_history = []
    avg_reward_history = []

    for episode in range(0, num_episodes, batch_size):
        trajectories = collect_batch(env, policy, device, batch_size, gamma)

        total_returns = [t['total_reward'] for t in trajectories]
        sum_returns = sum(total_returns)

        policy_loss = 0
        for i, traj in enumerate(trajectories):
            returns_tensor = torch.FloatTensor(traj['returns']).to(device)
            log_probs = traj['log_probs']
            if log_probs.dim() == 0:
                log_probs = log_probs.unsqueeze(0)

            if batch_size > 1:
                baseline = (sum_returns - total_returns[i]) / (batch_size - 1)
            else:
                baseline = 0.0

            advantage = returns_tensor - baseline
            policy_loss += -(log_probs * advantage).sum()

        policy_loss /= batch_size

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        batch_rewards = total_returns
        mean_reward = np.mean(batch_rewards)
        reward_history.extend(batch_rewards)
        avg_reward_history.append(mean_reward)

        if (episode // batch_size) % 20 == 0:
            print(f"[RLOO] Episode {episode}, Mean Reward: {mean_reward:.1f}")

    return policy, reward_history, avg_reward_history


def train_pg_entropy_reg(env, num_episodes=1000, batch_size=10, lr=1e-3,
                          gamma=0.99, entropy_coef_start=0.1, entropy_coef_end=0.001,
                          baseline_type='value', device=None, seed=42):

    torch.manual_seed(seed)
    policy = PolicyNetwork().to(device)
    value_net = ValueNetwork().to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)

    reward_history = []
    avg_reward_history = []
    total_updates = num_episodes // batch_size

    for update_idx, episode in enumerate(range(0, num_episodes, batch_size)):
        trajectories = collect_batch(env, policy, device, batch_size, gamma)

        fraction = update_idx / max(total_updates - 1, 1)
        entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * fraction

        policy_loss = 0
        value_loss = 0
        entropy_bonus = 0

        for traj in trajectories:
            states_tensor = torch.FloatTensor(traj['states']).to(device)
            returns_tensor = torch.FloatTensor(traj['returns']).to(device)
            log_probs = traj['log_probs']
            entropies = traj['entropies']
            if log_probs.dim() == 0:
                log_probs = log_probs.unsqueeze(0)
                entropies = entropies.unsqueeze(0)

            values = value_net(states_tensor)
            advantage = (returns_tensor - values.detach())

            policy_loss += -(log_probs * advantage).sum()
            value_loss += nn.functional.mse_loss(values, returns_tensor)
            entropy_bonus += entropies.sum()

        policy_loss /= batch_size
        value_loss /= batch_size
        entropy_bonus /= batch_size

        total_policy_loss = policy_loss - entropy_coef * entropy_bonus

        policy_optimizer.zero_grad()
        total_policy_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        batch_rewards = [t['total_reward'] for t in trajectories]
        mean_reward = np.mean(batch_rewards)
        reward_history.extend(batch_rewards)
        avg_reward_history.append(mean_reward)

        if update_idx % 20 == 0:
            print(f"[Entropy] Episode {episode}, Mean Reward: {mean_reward:.1f}, "
                  f"β={entropy_coef:.4f}")

    return policy, reward_history, avg_reward_history
