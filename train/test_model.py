from stable_baselines3 import DQN
from env.spectrum_env import SpectrumEnv

env = SpectrumEnv()

model = DQN.load("dqn_spectrum_model")

obs, _ = env.reset()

for step in range(20):

    action, _states = model.predict(obs)

    obs, reward, done, truncated, info = env.step(action)

    print(f"Step: {step}")
    print(f"Chosen Channel: {action}")
    print(f"Occupancy: {obs}")
    print(f"Reward: {reward}")
    print("-" * 30)