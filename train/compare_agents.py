import numpy as np
import matplotlib.pyplot as plt
from env.spectrum_env import SpectrumEnv
from stable_baselines3 import DQN, PPO
import os

os.makedirs("results", exist_ok=True)

env = SpectrumEnv()
dqn = DQN.load("./models/dqn_model")
ppo = PPO.load("./models/ppo_model")

NUM_EPISODES = 100

def evaluate(model, env):
    rewards, throughputs, sinrs = [], [], []
    for _ in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_reward, ep_tp, ep_sinr = 0, [], []
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_tp.append(info["throughput"])
            ep_sinr.append(info["sinr"])
        rewards.append(ep_reward)
        throughputs.append(np.mean(ep_tp))
        sinrs.append(np.mean(ep_sinr))
    return np.mean(rewards), np.mean(throughputs), np.mean(sinrs)

dqn_r, dqn_tp, dqn_sinr = evaluate(dqn, env)
ppo_r, ppo_tp, ppo_sinr = evaluate(ppo, env)

print(f"\n===== AGENT COMPARISON =====")
print(f"DQN  — Reward: {dqn_r:.2f}  Throughput: {dqn_tp:.4f}  SINR: {dqn_sinr:.4f}")
print(f"PPO  — Reward: {ppo_r:.2f}  Throughput: {ppo_tp:.4f}  SINR: {ppo_sinr:.4f}")

metrics = ["Avg Reward", "Avg Throughput", "Avg SINR"]
dqn_vals = [dqn_r, dqn_tp * 100, dqn_sinr * 10]
ppo_vals = [ppo_r, ppo_tp * 100, ppo_sinr * 10]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, dqn_vals, width, label="DQN")
bars2 = ax.bar(x + width/2, ppo_vals, width, label="PPO")

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title("DQN vs PPO Performance Comparison")
ax.set_ylabel("Score (scaled for display)")
ax.legend()
ax.grid(True, axis='y')
plt.tight_layout()
plt.savefig("results/dqn_vs_ppo.png", dpi=300, bbox_inches='tight')
print("Comparison plot saved.")
plt.show(block=True)