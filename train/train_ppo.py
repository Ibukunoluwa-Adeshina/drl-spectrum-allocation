from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env.spectrum_env import SpectrumEnv

import os

# create results directory
os.makedirs("results", exist_ok=True)

# environment
env = SpectrumEnv()

# monitor wrapper
env = Monitor(
    env,
    filename="results/ppo_monitor.csv"
)

# PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

# train model
model.learn(
    total_timesteps=20000
)

# save model
model.save(
    "results/ppo_spectrum_model"
)

print("PPO training complete.")