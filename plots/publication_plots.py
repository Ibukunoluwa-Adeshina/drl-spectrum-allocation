import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

algorithms = ["DQN", "PPO"]

rewards = [644.20, 654.64]
throughput = [0.6698, 0.6705]
sinr = [3.2365, 3.2907]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(algorithms, rewards)

ax.set_title(
    "Average Reward Comparison",
    fontsize=18,
    fontweight='bold'
)

ax.set_ylabel("Average Reward", fontsize=14)
ax.set_xlabel("Algorithm", fontsize=14)

for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        yval + 2,
        round(yval, 2),
        ha='center',
        fontsize=12
    )

plt.tight_layout()

plt.savefig(
    "outputs/figures/reward_comparison.png",
    dpi=300
)

plt.show()