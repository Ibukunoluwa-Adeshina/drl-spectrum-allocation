import numpy as np
import matplotlib.pyplot as plt
from env.spectrum_env import SpectrumEnv
from stable_baselines3 import DQN
import os

# create results folder
os.makedirs("results", exist_ok=True)

# environment
env = SpectrumEnv()

# load trained model
model = DQN.load("results/dqn_spectrum_model")

NUM_EPISODES = 100

high_priority_rewards = []
low_priority_rewards = []

# evaluation loop
for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False

    high_rewards = []
    low_rewards = []

    while not done:

        action, _ = model.predict(obs)

        obs, reward, done, truncated, info = env.step(action)

        if info["traffic_type"] == "high":
            high_rewards.append(reward)

        else:
            low_rewards.append(reward)

    if len(high_rewards) > 0:
        high_priority_rewards.append(
            np.mean(high_rewards)
        )

    if len(low_rewards) > 0:
        low_priority_rewards.append(
            np.mean(low_rewards)
        )

# averages
avg_high = np.mean(high_priority_rewards)
avg_low = np.mean(low_priority_rewards)

print("\n===== QoS ANALYSIS =====")
print(f"High Priority Reward: {avg_high:.2f}")
print(f"Low Priority Reward: {avg_low:.2f}")

# plot
labels = [
    "High Priority",
    "Low Priority"
]

values = [
    avg_high,
    avg_low
]

plt.figure(figsize=(8,5))

bars = plt.bar(labels, values)

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
plt.title("QoS-Aware Scheduling Performance")

plt.grid(True)

# save figure
plt.savefig(
    "results/qos_analysis.png",
    dpi=300,
    bbox_inches='tight'
)

print("QoS plot saved.")

plt.show(block=True)