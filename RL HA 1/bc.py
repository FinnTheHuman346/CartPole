import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from config import DEVICE, SEED
from models import PolicyNetwork
from eval_utils import evaluate_policy


def generate_expert_dataset(env, expert_policy, device, num_trajectories=200,
                             min_reward=450):
    """
    Генерирует датасет экспертных траекторий.
    Берём только те, где reward >= min_reward (фильтрация).
    """
    states_all = []
    actions_all = []
    total_collected = 0
    total_attempts = 0

    while total_collected < num_trajectories:
        state, _ = env.reset()
        trajectory_states = []
        trajectory_actions = []
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                dist = expert_policy.get_distribution(state_tensor)
                action = dist.sample().item()

            trajectory_states.append(state)
            trajectory_actions.append(action)

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        total_attempts += 1

        if total_reward >= min_reward:
            states_all.extend(trajectory_states)
            actions_all.extend(trajectory_actions)
            total_collected += 1

    print(f"Collected {total_collected} trajectories "
          f"({total_attempts} attempts, "
          f"{len(states_all)} state-action pairs)")

    return np.array(states_all), np.array(actions_all)


def train_behaviour_cloning(states, actions, num_epochs=100, lr=1e-3,
                              batch_size=256, device=DEVICE, seed=SEED):
    """
    Обучает политику через Behaviour Cloning (supervised learning).
    Loss = CrossEntropy(policy(s), a_expert)
    """
    torch.manual_seed(seed)
    policy = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)

    dataset_size = len(states)
    loss_history = []

    for epoch in range(num_epochs):
        perm = torch.randperm(dataset_size)
        states_shuffled = states_tensor[perm]
        actions_shuffled = actions_tensor[perm]

        epoch_loss = 0
        num_batches = 0

        for i in range(0, dataset_size, batch_size):
            batch_states = states_shuffled[i:i+batch_size]
            batch_actions = actions_shuffled[i:i+batch_size]

            logits = policy(batch_states)
            loss = nn.functional.cross_entropy(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        if epoch % 20 == 0:
            print(f"  [BC] Epoch {epoch}, Loss: {avg_loss:.4f}")

    return policy, loss_history


def run_bc_experiments(expert_policy):
    """Запускает все эксперименты по Behaviour Cloning."""

    env = gym.make("CartPole-v1")

    print("=" * 60)
    print("BEHAVIOUR CLONING EXPERIMENTS")
    print("=" * 60)

    expert_mean, expert_std, _ = evaluate_policy(env, expert_policy, DEVICE, 100)
    print(f"Expert performance: {expert_mean:.1f} ± {expert_std:.1f}")

    print("\n--- Generating expert dataset ---")
    threshold = 450 if expert_mean > 450 else expert_mean * 0.8
    states, actions = generate_expert_dataset(
        env, expert_policy, DEVICE,
        num_trajectories=200, min_reward=threshold
    )

    print("\n--- Training BC policy ---")
    bc_policy, bc_losses = train_behaviour_cloning(
        states, actions, num_epochs=100, lr=1e-3
    )

    bc_mean, bc_std, _ = evaluate_policy(env, bc_policy, DEVICE, 100)
    print(f"\nBC policy performance: {bc_mean:.1f} ± {bc_std:.1f}")
    print(f"Expert performance:    {expert_mean:.1f} ± {expert_std:.1f}")

    plt.figure(figsize=(8, 4))
    plt.plot(bc_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Behaviour Cloning Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bc_training_loss.png", dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("BC FAILURE EXPERIMENTS")
    print("=" * 60)

    print("\n--- Experiment 1: Effect of dataset size ---")
    sizes = [10, 50, 100, 200]
    size_results = {}

    for n_traj in sizes:
        s, a = generate_expert_dataset(
            env, expert_policy, DEVICE,
            num_trajectories=n_traj, min_reward=threshold
        )
        bc_pol, _ = train_behaviour_cloning(s, a, num_epochs=100, lr=1e-3)
        mean_r, std_r, _ = evaluate_policy(env, bc_pol, DEVICE, 50)
        size_results[n_traj] = (mean_r, std_r)
        print(f"  {n_traj} trajectories: {mean_r:.1f} ± {std_r:.1f}")

    plt.figure(figsize=(8, 4))
    sizes_list = list(size_results.keys())
    means = [size_results[s][0] for s in sizes_list]
    stds = [size_results[s][1] for s in sizes_list]
    plt.errorbar(sizes_list, means, yerr=stds, marker='o', capsize=5)
    plt.xlabel("Number of Expert Trajectories")
    plt.ylabel("Mean Reward")
    plt.title("BC Performance vs Dataset Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bc_dataset_size.png", dpi=150)
    plt.show()

    print("\n--- Experiment 2: Noisy/suboptimal expert ---")
    noise_results = {}

    for noise_prob in [0.0, 0.05, 0.1, 0.2, 0.3]:
        noisy_states, noisy_actions = [], []
        for _ in range(100):
            state, _ = env.reset()
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    dist = expert_policy.get_distribution(state_tensor)
                    action = dist.sample().item()

                if np.random.random() < noise_prob:
                    action = env.action_space.sample()

                noisy_states.append(state)
                noisy_actions.append(action)

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

        noisy_states = np.array(noisy_states)
        noisy_actions = np.array(noisy_actions)

        bc_pol, _ = train_behaviour_cloning(
            noisy_states, noisy_actions, num_epochs=100, lr=1e-3
        )
        mean_r, std_r, _ = evaluate_policy(env, bc_pol, DEVICE, 50)
        noise_results[noise_prob] = (mean_r, std_r)
        print(f"  Noise {noise_prob:.0%}: {mean_r:.1f} ± {std_r:.1f}")

    plt.figure(figsize=(8, 4))
    noise_list = list(noise_results.keys())
    means = [noise_results[n][0] for n in noise_list]
    stds = [noise_results[n][1] for n in noise_list]
    plt.errorbar(noise_list, means, yerr=stds, marker='o', capsize=5)
    plt.xlabel("Expert Noise Probability")
    plt.ylabel("Mean Reward")
    plt.title("BC Performance with Noisy Expert")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bc_noisy_expert.png", dpi=150)
    plt.show()

    print("\n--- Experiment 3: Distribution shift analysis ---")
    print("  Comparing state distributions visited by expert vs BC policy...")

    bc_states_visited = []
    expert_states_visited = []

    for _ in range(50):
        state, _ = env.reset()
        done = False
        while not done:
            expert_states_visited.append(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = expert_policy.get_distribution(state_tensor).sample().item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        state, _ = env.reset()
        done = False
        while not done:
            bc_states_visited.append(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = bc_policy.get_distribution(state_tensor).sample().item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    expert_states_visited = np.array(expert_states_visited)
    bc_states_visited = np.array(bc_states_visited)

    feature_names = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
        ax.hist(expert_states_visited[:, i], bins=50, alpha=0.5, label='Expert', density=True)
        ax.hist(bc_states_visited[:, i], bins=50, alpha=0.5, label='BC', density=True)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("State Distribution: Expert vs BC Policy")
    plt.tight_layout()
    plt.savefig("bc_distribution_shift.png", dpi=150)
    plt.show()

    env.close()
    return bc_policy
