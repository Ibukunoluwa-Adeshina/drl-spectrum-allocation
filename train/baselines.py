import numpy as np
from env.spectrum_env import SpectrumEnv
from stable_baselines3 import DQN

env = SpectrumEnv()
model = DQN.load("./models/dqn_model")

NUM_EPISODES = 100

# =========================
# RANDOM POLICY
# =========================
random_rewards = []
random_throughputs = []
random_sinrs = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    ep_tp, ep_sinr = [], []

    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        ep_tp.append(info["throughput"])
        ep_sinr.append(info["sinr"])

    random_rewards.append(total_reward)
    random_throughputs.append(np.mean(ep_tp))
    random_sinrs.append(np.mean(ep_sinr))

# =========================
# GREEDY POLICY
# =========================
greedy_rewards = []
greedy_throughputs = []
greedy_sinrs = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    ep_tp, ep_sinr = [], []

    while not done:
        action = np.argmin(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        ep_tp.append(info["throughput"])
        ep_sinr.append(info["sinr"])

    greedy_rewards.append(total_reward)
    greedy_throughputs.append(np.mean(ep_tp))
    greedy_sinrs.append(np.mean(ep_sinr))

# =========================
# DQN POLICY
# =========================
dqn_rewards = []
dqn_throughputs = []
dqn_sinrs = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    ep_tp, ep_sinr = [], []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        ep_tp.append(info["throughput"])
        ep_sinr.append(info["sinr"])

    dqn_rewards.append(total_reward)
    dqn_throughputs.append(np.mean(ep_tp))
    dqn_sinrs.append(np.mean(ep_sinr))

# =========================
# RESULTS
# =========================
print("\n========== FULL BASELINE RESULTS ==========")
print(f"{'Method':<10} {'Avg Reward':>12} {'Avg Throughput':>16} {'Avg SINR':>12}")
print("-" * 54)
print(f"{'Random':<10} {np.mean(random_rewards):>12.2f} {np.mean(random_throughputs):>16.4f} {np.mean(random_sinrs):>12.4f}")
print(f"{'Greedy':<10} {np.mean(greedy_rewards):>12.2f} {np.mean(greedy_throughputs):>16.4f} {np.mean(greedy_sinrs):>12.4f}")
print(f"{'DQN':<10} {np.mean(dqn_rewards):>12.2f} {np.mean(dqn_throughputs):>16.4f} {np.mean(dqn_sinrs):>12.4f}")
print("=" * 54)