import gymnasium as gym
import numpy as np

from config import DEVICE, SEED
from pg_trainers import (
    train_vanilla_pg,
    train_pg_average_baseline,
    train_pg_value_baseline,
    train_pg_rloo_baseline,
    train_pg_entropy_reg,
)
from plot_utils import plot_comparison, plot_batch_comparison
from eval_utils import evaluate_policy


def run_all_rl_experiments():

    env = gym.make("CartPole-v1")

    NUM_EPISODES = 2000
    BATCH_SIZE = 10
    LR = 1e-3
    GAMMA = 0.99

    print("=" * 60)
    print("1. Vanilla Policy Gradient")
    print("=" * 60)
    policy_vpg, hist_vpg, avg_vpg = train_vanilla_pg(
        env, NUM_EPISODES, BATCH_SIZE, LR, GAMMA, device=DEVICE, seed=SEED
    )

    print("\n" + "=" * 60)
    print("2. PG + Average Reward Baseline")
    print("=" * 60)
    policy_avg, hist_avg, avg_avg = train_pg_average_baseline(
        env, NUM_EPISODES, BATCH_SIZE, LR, GAMMA, device=DEVICE, seed=SEED
    )

    print("\n" + "=" * 60)
    print("3. PG + Value Function Baseline")
    print("=" * 60)
    policy_val, hist_val, avg_val = train_pg_value_baseline(
        env, NUM_EPISODES, BATCH_SIZE, LR, LR, GAMMA, device=DEVICE, seed=SEED
    )

    print("\n" + "=" * 60)
    print("4. PG + RLOO Baseline")
    print("=" * 60)
    policy_rloo, hist_rloo, avg_rloo = train_pg_rloo_baseline(
        env, NUM_EPISODES, BATCH_SIZE, LR, GAMMA, device=DEVICE, seed=SEED
    )

    print("\n" + "=" * 60)
    print("5. PG + Value Baseline + Entropy Regularization")
    print("=" * 60)
    policy_ent, hist_ent, avg_ent = train_pg_entropy_reg(
        env, NUM_EPISODES, BATCH_SIZE, LR, GAMMA,
        entropy_coef_start=0.1, entropy_coef_end=0.001,
        device=DEVICE, seed=SEED
    )

    results_raw = {
        "Vanilla PG": hist_vpg,
        "PG + Avg Baseline": hist_avg,
        "PG + Value Baseline": hist_val,
        "PG + RLOO Baseline": hist_rloo,
        "PG + Value + Entropy": hist_ent,
    }
    plot_comparison(results_raw, filename="rl_comparison_raw.png")

    results_batch = {
        "Vanilla PG": avg_vpg,
        "PG + Avg Baseline": avg_avg,
        "PG + Value Baseline": avg_val,
        "PG + RLOO Baseline": avg_rloo,
        "PG + Value + Entropy": avg_ent,
    }
    plot_batch_comparison(results_batch, filename="rl_comparison_batch.png")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION (100 episodes each)")
    print("=" * 60)

    policies = {
        "Vanilla PG": policy_vpg,
        "PG + Avg Baseline": policy_avg,
        "PG + Value Baseline": policy_val,
        "PG + RLOO Baseline": policy_rloo,
        "PG + Value + Entropy": policy_ent,
    }

    best_name, best_mean = None, 0
    for name, pol in policies.items():
        mean_r, std_r, _ = evaluate_policy(env, pol, DEVICE, 100)
        print(f"  {name}: {mean_r:.1f} ± {std_r:.1f}")
        if mean_r > best_mean:
            best_mean = mean_r
            best_name = name

    print(f"\nBest policy: {best_name} with mean reward {best_mean:.1f}")

    env.close()
    return policies, results_raw, results_batch, best_name
