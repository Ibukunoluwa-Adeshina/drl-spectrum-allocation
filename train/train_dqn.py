from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from env.spectrum_env import SpectrumEnv
import os

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/logs/dqn", exist_ok=True)

env = SpectrumEnv()
env = Monitor(env, filename="results/monitor.csv")

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    buffer_size=10000,
    learning_starts=100,
    batch_size=32,
    gamma=0.95,
    tensorboard_log="./outputs/logs/dqn/"
)

model.learn(total_timesteps=20000)
model.save("./models/dqn_model")

print("DQN training complete.")