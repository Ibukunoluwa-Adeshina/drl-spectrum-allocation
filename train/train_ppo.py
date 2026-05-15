from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.spectrum_env import SpectrumEnv
import os

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/logs/ppo", exist_ok=True)

env = SpectrumEnv()
env = Monitor(env, filename="results/ppo_monitor.csv")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./outputs/logs/ppo/"
)

model.learn(total_timesteps=20000)
model.save("./models/ppo_model")

print("PPO training complete.")