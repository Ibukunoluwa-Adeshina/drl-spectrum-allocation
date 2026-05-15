import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.spectrum_env import SpectrumEnv

import os

# create results directory
os.makedirs("results", exist_ok=True)

# environment
env = SpectrumEnv()

# load PPO model
model = PPO.load(
    "results/ppo_spectrum_model"
)

NUM_EPISODES = 100

rewards = []
throughputs = []
sinr_values = []

# evaluation loop
for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False

    episode_reward = 0
    episode_throughput = []
    episode_sinr = []

    while not done:

        action, _ = model.predict(obs)

        obs, reward, done, truncated, info = env.step(action)

        episode_reward += reward

        episode_throughput.append(
            info["throughput"]
        )

        episode_sinr.append(
            info["sinr"]
        )

    rewards.append(
        episode_reward
    )

    throughputs.append(
        np.mean(episode_throughput)
    )

    sinr_values.append(
        np.mean(episode_sinr)
    )

print("\n===== PPO PERFORMANCE =====")

print(f"Average Reward: {np.mean(rewards):.2f}")

print(f"Average Throughput: {np.mean(throughputs):.4f}")

print(f"Average SINR: {np.mean(sinr_values):.4f}")

# reward plot
plt.figure(figsize=(10,5))

plt.plot(
    rewards,
    label="PPO Rewards"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO Reward Performance")

plt.grid(True)
plt.legend()

plt.savefig(
    "results/ppo_rewards.png",
    dpi=300,
    bbox_inches='tight'
)

print("PPO reward plot saved.")

plt.show(block=True)