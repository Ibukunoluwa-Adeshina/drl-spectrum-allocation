from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from env.spectrum_env import SpectrumEnv
import os

# create results folder
os.makedirs("results", exist_ok=True)

# create environment
env = SpectrumEnv()

# add monitor wrapper
env = Monitor(env, filename="results/monitor.csv")

# create DQN model
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    buffer_size=10000,
    learning_starts=100,
    batch_size=32,
    gamma=0.95
)

# train
model.learn(total_timesteps=20000)

# save model
model.save("results/dqn_spectrum_model")

print("Training complete.")