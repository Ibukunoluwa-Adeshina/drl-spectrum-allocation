import numpy as np
import matplotlib.pyplot as plt
from env.spectrum_env import SpectrumEnv
from stable_baselines3 import DQN
import os

# create results folder
os.makedirs("results", exist_ok=True)

# environment
env = SpectrumEnv()

# load model
model = DQN.load(
    "results/dqn_spectrum_model"
)

NUM_EPISODES = 100

distances = []
sinr_values = []

# evaluation loop
for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False

    while not done:

        action, _ = model.predict(obs)

        obs, reward, done, truncated, info = env.step(action)

        distances.append(
            info["distance"]
        )

        sinr_values.append(
            info["sinr"]
        )

# =========================
# PLOT
# =========================

plt.figure(figsize=(10,5))

plt.scatter(
    distances,
    sinr_values,
    alpha=0.6
)

plt.xlabel("Distance from Base Station")
plt.ylabel("SINR")
plt.title("SINR vs User Distance")

plt.grid(True)

# save
plt.savefig(
    "results/mobility_analysis.png",
    dpi=300,
    bbox_inches='tight'
)

print("Mobility plot saved.")

plt.show(block=True)