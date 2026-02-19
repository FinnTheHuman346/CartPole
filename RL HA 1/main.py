from config import DEVICE
from rl_experiments import run_all_rl_experiments
from bc import run_bc_experiments

if __name__ == "__main__":
    print("=" * 60)
    print("  PART 1: REINFORCEMENT LEARNING")
    print("=" * 60)
    policies, results_raw, results_batch, best_name = run_all_rl_experiments()

    print("\n\n" + "=" * 60)
    print("  PART 2: BEHAVIOUR CLONING")
    print("=" * 60)
    best_policy = policies[best_name]
    bc_policy = run_bc_experiments(best_policy)

    print("\n\nDone! All plots saved.")
