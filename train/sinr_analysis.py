import numpy as np
import matplotlib.pyplot as plt
from env.spectrum_env import SpectrumEnv
from stable_baselines3 import DQN
import os

# create results directory
os.makedirs("results", exist_ok=True)

# environment
env = SpectrumEnv()

# load trained model
model = DQN.load("results/dqn_spectrum_model")

NUM_EPISODES = 100

sinr_values = []

# evaluation loop
for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False

    episode_sinr = []

    while not done:

        # DQN prediction
        action, _ = model.predict(obs)

        # environment step
        obs, reward, done, truncated, info = env.step(action)

        # collect SINR
        episode_sinr.append(
            info["sinr"]
        )

    # average SINR per episode
    avg_sinr = np.mean(
        episode_sinr
    )

    sinr_values.append(
        avg_sinr
    )

# overall average SINR
overall_sinr = np.mean(
    sinr_values
)

print("\n===== SINR ANALYSIS =====")
print(f"Average SINR: {overall_sinr:.4f}")

# =========================
# PLOT
# =========================

plt.figure(figsize=(10,5))

plt.plot(
    sinr_values,
    label="Episode SINR"
)

plt.xlabel("Episode")
plt.ylabel("Average SINR")
plt.title("DQN SINR Performance")

plt.grid(True)
plt.legend()

# save figure
plt.savefig(
    "results/sinr_analysis.png",
    dpi=300,
    bbox_inches='tight'
)

print("SINR plot saved.")

plt.show(block=True)