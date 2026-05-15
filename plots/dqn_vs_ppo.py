import matplotlib.pyplot as plt
import os

# create results directory
os.makedirs("results", exist_ok=True)

# YOUR ACTUAL RESULTS
algorithms = [
    "DQN",
    "PPO"
]

rewards = [
    644.20,   # replace with YOUR DQN reward
    654.64    # replace with YOUR PPO reward
]

throughputs = [
    0.6698,     # replace with YOUR DQN throughput
    0.6705      # replace with YOUR PPO throughput
]

sinr_values = [
    3.2365,     # replace with YOUR DQN SINR
    3.2907      # replace with YOUR PPO SINR
]

# =========================
# REWARD COMPARISON
# =========================

plt.figure(figsize=(8,5))

bars = plt.bar(
    algorithms,
    rewards
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom'
    )

plt.ylabel("Average Reward")
plt.title("DQN vs PPO Reward Comparison")

plt.grid(True)

plt.savefig(
    "results/dqn_vs_ppo_rewards.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show(block=True)

# =========================
# THROUGHPUT COMPARISON
# =========================

plt.figure(figsize=(8,5))

bars = plt.bar(
    algorithms,
    throughputs
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom'
    )

plt.ylabel("Average Throughput")
plt.title("DQN vs PPO Throughput Comparison")

plt.grid(True)

plt.savefig(
    "results/dqn_vs_ppo_throughput.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show(block=True)

# =========================
# SINR COMPARISON
# =========================

plt.figure(figsize=(8,5))

bars = plt.bar(
    algorithms,
    sinr_values
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom'
    )

plt.ylabel("Average SINR")
plt.title("DQN vs PPO SINR Comparison")

plt.grid(True)

plt.savefig(
    "results/dqn_vs_ppo_sinr.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show(block=True)

print("DQN vs PPO comparison plots saved.")