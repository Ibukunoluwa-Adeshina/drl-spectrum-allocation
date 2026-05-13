import numpy as np
from env.spectrum_env import SpectrumEnv
from stable_baselines3 import DQN

# environment
env = SpectrumEnv()

# load trained RL model
model = DQN.load("results/dqn_spectrum_model")

NUM_EPISODES = 100

# =========================
# RANDOM POLICY
# =========================

random_rewards = []

for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward

    random_rewards.append(total_reward)

# =========================
# GREEDY POLICY
# =========================

greedy_rewards = []

for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        # choose least occupied channel
        action = np.argmin(obs)

        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward

    greedy_rewards.append(total_reward)

# =========================
# DQN POLICY
# =========================

dqn_rewards = []

for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        action, _ = model.predict(obs)

        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward

    dqn_rewards.append(total_reward)

# =========================
# RESULTS
# =========================

print("\n===== AVERAGE REWARDS =====")

print(f"Random Policy: {np.mean(random_rewards):.2f}")

print(f"Greedy Policy: {np.mean(greedy_rewards):.2f}")

print(f"DQN Policy: {np.mean(dqn_rewards):.2f}")