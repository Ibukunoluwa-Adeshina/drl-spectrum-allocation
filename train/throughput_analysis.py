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

throughputs = []

# run evaluation
for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False

    episode_throughput = []

    while not done:

        action, _ = model.predict(obs)

        obs, reward, done, truncated, info = env.step(action)

        episode_throughput.append(
            info["throughput"]
        )

    avg_episode_throughput = np.mean(
        episode_throughput
    )

    throughputs.append(
        avg_episode_throughput
    )

# overall average
overall_avg = np.mean(throughputs)

print("\n===== THROUGHPUT ANALYSIS =====")
print(f"Average Throughput: {overall_avg:.4f}")

# plot
plt.figure(figsize=(10,5))

plt.plot(
    throughputs,
    label="Episode Throughput"
)

plt.xlabel("Episode")
plt.ylabel("Average Throughput")
plt.title("DQN Throughput Performance")

plt.grid(True)
plt.legend()

# save
plt.savefig(
    "results/throughput_analysis.png",
    dpi=300,
    bbox_inches='tight'
)

print("Throughput plot saved.")

plt.show(block=True)