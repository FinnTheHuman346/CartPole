import numpy as np
import matplotlib.pyplot as plt

def smooth(data, window=20):
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed


def plot_comparison(results, title="Comparison of PG Methods", filename="comparison.png"):

    plt.figure(figsize=(12, 6))

    for name, rewards in results.items():
        smoothed = smooth(rewards, window=50)
        plt.plot(smoothed, label=name, alpha=0.9)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward (smoothed)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Plot saved to {filename}")


def plot_batch_comparison(results, title="Comparison (batch avg)", filename="comparison_batch.png"):
    plt.figure(figsize=(12, 6))

    for name, avg_rewards in results.items():
        smoothed = smooth(avg_rewards, window=10)
        plt.plot(smoothed, label=name, alpha=0.9)

    plt.xlabel("Batch Update")
    plt.ylabel("Mean Batch Reward (smoothed)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Plot saved to {filename}")
