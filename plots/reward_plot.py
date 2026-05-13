import pandas as pd
import matplotlib.pyplot as plt

# load monitor file
df = pd.read_csv(
    "results/monitor.csv",
    skiprows=1
)

# rewards
rewards = df["r"]

# moving average
window = 20
moving_avg = rewards.rolling(window).mean()

# create figure
plt.figure(figsize=(10,5))

# plot lines
plt.plot(rewards, label="Episode Reward")
plt.plot(moving_avg, label="Moving Average")

# labels
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Spectrum Allocation Learning Curve")

# extras
plt.legend()
plt.grid(True)

# SAVE PLOT
plt.savefig("results/reward_plot.png")

# SHOW PLOT
plt.show(block=True)