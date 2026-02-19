import numpy as np
import torch

def collect_trajectory(env, policy, device, gamma=0.99):
    states, actions, rewards, log_probs, entropies = [], [], [], [], []

    state, _ = env.reset()
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, entropy = policy.select_action(state_tensor)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        entropies.append(entropy)

        state = next_state

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'returns': np.array(returns),
        'log_probs': torch.stack(log_probs).squeeze(),
        'entropies': torch.stack(entropies).squeeze(),
        'total_reward': sum(rewards),
    }


def collect_batch(env, policy, device, batch_size=10, gamma=0.99):
    trajectories = []
    for _ in range(batch_size):
        traj = collect_trajectory(env, policy, device, gamma)
        trajectories.append(traj)
    return trajectories
