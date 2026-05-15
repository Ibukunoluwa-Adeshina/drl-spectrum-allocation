import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import DQN, PPO
from env.spectrum_env import SpectrumEnv
import os

os.makedirs("results/publication", exist_ok=True)

# ── consistent style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DQN_COLOR = "#185FA5"
PPO_COLOR = "#D85A30"
HIGH_COLOR = "#1D9E75"
LOW_COLOR  = "#D85A30"
NUM_EPISODES = 200

# ── load models & env ──────────────────────────────────────────────────────────
env = SpectrumEnv()
dqn = DQN.load("./models/dqn_model")
ppo = PPO.load("./models/ppo_model")

# ── helper: run full evaluation ────────────────────────────────────────────────
def run_eval(model, n=NUM_EPISODES):
    rewards, throughputs, sinrs, distances = [], [], [], []
    high_r, low_r = [], []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        ep_r, ep_tp, ep_sinr, ep_dist = 0, [], [], []
        ep_high, ep_low = [], []
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            ep_r += reward
            ep_tp.append(info["throughput"])
            ep_sinr.append(info["sinr"])
            ep_dist.append(info["distance"])
            if info["traffic_type"] == "high":
                ep_high.append(reward)
            else:
                ep_low.append(reward)
        rewards.append(ep_r)
        throughputs.append(np.mean(ep_tp))
        sinrs.append(np.mean(ep_sinr))
        distances.append(np.mean(ep_dist))
        if ep_high: high_r.append(np.mean(ep_high))
        if ep_low:  low_r.append(np.mean(ep_low))
    return dict(
        rewards=rewards, throughputs=throughputs, sinrs=sinrs,
        distances=distances, high_r=high_r, low_r=low_r
    )

print("Evaluating DQN …")
dqn_data = run_eval(dqn)
print("Evaluating PPO …")
ppo_data = run_eval(ppo)

# ── load monitor CSVs for learning curves ─────────────────────────────────────
def load_monitor(path):
    try:
        df = pd.read_csv(path, skiprows=1)
        return df["r"].values
    except Exception:
        return np.array([])

dqn_curve = load_monitor("results/monitor.csv")
ppo_curve = load_monitor("results/ppo_monitor.csv")

def moving_avg(arr, w=20):
    if len(arr) < w:
        return arr
    return pd.Series(arr).rolling(w).mean().values


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Learning curves (DQN & PPO side by side)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, curve, color, label in zip(
    axes,
    [dqn_curve, ppo_curve],
    [DQN_COLOR, PPO_COLOR],
    ["DQN", "PPO"]
):
    if len(curve) == 0:
        ax.text(0.5, 0.5, f"No monitor data\nfor {label}",
                ha="center", va="center", transform=ax.transAxes)
    else:
        episodes = np.arange(1, len(curve) + 1)
        ax.plot(episodes, curve, color=color, alpha=0.25, linewidth=0.8, label="Episode reward")
        ma = moving_avg(curve, 20)
        ax.plot(episodes, ma, color=color, linewidth=2, label="Moving avg (20)")
        ax.set_title(f"{label} Learning Curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend()

fig.suptitle("Training Learning Curves — DQN vs PPO", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/publication/plot1_learning_curves.png", dpi=300, bbox_inches="tight")
print("Saved: plot1_learning_curves.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Reward distribution (histogram + KDE overlay)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

for data, color, label in [
    (dqn_data["rewards"], DQN_COLOR, "DQN"),
    (ppo_data["rewards"], PPO_COLOR, "PPO"),
]:
    ax.hist(data, bins=30, color=color, alpha=0.45, density=True, label=label)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 300)
    ax.plot(xs, kde(xs), color=color, linewidth=2)
    ax.axvline(np.mean(data), color=color, linestyle="--", linewidth=1.2,
               label=f"{label} mean={np.mean(data):.1f}")

ax.set_xlabel("Episode Reward")
ax.set_ylabel("Density")
ax.set_title("Reward Distribution — DQN vs PPO")
ax.legend()
plt.tight_layout()
plt.savefig("results/publication/plot2_reward_distribution.png", dpi=300, bbox_inches="tight")
print("Saved: plot2_reward_distribution.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Throughput distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

for data, color, label in [
    (dqn_data["throughputs"], DQN_COLOR, "DQN"),
    (ppo_data["throughputs"], PPO_COLOR, "PPO"),
]:
    ax.hist(data, bins=30, color=color, alpha=0.45, density=True, label=label)
    kde = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 300)
    ax.plot(xs, kde(xs), color=color, linewidth=2)
    ax.axvline(np.mean(data), color=color, linestyle="--", linewidth=1.2,
               label=f"{label} mean={np.mean(data):.4f}")

ax.set_xlabel("Average Episode Throughput")
ax.set_ylabel("Density")
ax.set_title("Throughput Distribution — DQN vs PPO")
ax.legend()
plt.tight_layout()
plt.savefig("results/publication/plot3_throughput_distribution.png", dpi=300, bbox_inches="tight")
print("Saved: plot3_throughput_distribution.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — SINR distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

for data, color, label in [
    (dqn_data["sinrs"], DQN_COLOR, "DQN"),
    (ppo_data["sinrs"], PPO_COLOR, "PPO"),
]:
    ax.hist(data, bins=30, color=color, alpha=0.45, density=True, label=label)
    kde = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 300)
    ax.plot(xs, kde(xs), color=color, linewidth=2)
    ax.axvline(np.mean(data), color=color, linestyle="--", linewidth=1.2,
               label=f"{label} mean={np.mean(data):.4f}")

ax.set_xlabel("Average Episode SINR")
ax.set_ylabel("Density")
ax.set_title("SINR Distribution — DQN vs PPO")
ax.legend()
plt.tight_layout()
plt.savefig("results/publication/plot4_sinr_distribution.png", dpi=300, bbox_inches="tight")
print("Saved: plot4_sinr_distribution.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Distance vs SINR scatter (mobility analysis)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

# re-collect step-level data for scatter
def collect_step_data(model, n=50):
    dist_arr, sinr_arr = [], []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _, info = env.step(action)
            dist_arr.append(info["distance"])
            sinr_arr.append(info["sinr"])
    return np.array(dist_arr), np.array(sinr_arr)

dqn_dist, dqn_sinr_step = collect_step_data(dqn)
ppo_dist, ppo_sinr_step = collect_step_data(ppo)

ax.scatter(dqn_dist, dqn_sinr_step, color=DQN_COLOR, alpha=0.25, s=8, label="DQN")
ax.scatter(ppo_dist, ppo_sinr_step, color=PPO_COLOR, alpha=0.25, s=8, label="PPO")

# trend lines
for dist, sinr, color in [(dqn_dist, dqn_sinr_step, DQN_COLOR),
                           (ppo_dist, ppo_sinr_step, PPO_COLOR)]:
    z = np.polyfit(dist, sinr, 1)
    p = np.poly1d(z)
    xs = np.linspace(dist.min(), dist.max(), 200)
    ax.plot(xs, p(xs), color=color, linewidth=2)

ax.set_xlabel("Distance from Base Station (m)")
ax.set_ylabel("SINR")
ax.set_title("SINR vs User Distance from Base Station")
ax.legend()
plt.tight_layout()
plt.savefig("results/publication/plot5_distance_vs_sinr.png", dpi=300, bbox_inches="tight")
print("Saved: plot5_distance_vs_sinr.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — QoS high vs low priority (both agents)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

labels  = ["DQN\nHigh Priority", "DQN\nLow Priority",
           "PPO\nHigh Priority", "PPO\nLow Priority"]
values  = [
    np.mean(dqn_data["high_r"]),
    np.mean(dqn_data["low_r"]),
    np.mean(ppo_data["high_r"]),
    np.mean(ppo_data["low_r"]),
]
colors  = [DQN_COLOR, DQN_COLOR, PPO_COLOR, PPO_COLOR]
alphas  = [1.0, 0.45, 1.0, 0.45]
hatches = ["", "//", "", "//"]

bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
for bar, a, h in zip(bars, alphas, hatches):
    bar.set_alpha(a)
    bar.set_hatch(h)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Average Reward")
ax.set_title("QoS-Aware Scheduling: High vs Low Priority Traffic")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=DQN_COLOR, label="DQN"),
    Patch(facecolor=PPO_COLOR, label="PPO"),
    Patch(facecolor="gray", alpha=0.45, hatch="//", label="Low priority"),
]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig("results/publication/plot6_qos_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: plot6_qos_comparison.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Summary dashboard (all 4 metrics in one figure)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(12, 8))
gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

metric_configs = [
    ("Average Episode Reward",      "rewards",     "{:.1f}"),
    ("Average Throughput",          "throughputs", "{:.4f}"),
    ("Average SINR",                "sinrs",       "{:.4f}"),
    ("High-Priority Avg Reward",    None,          "{:.2f}"),
]

for idx, (title, key, fmt) in enumerate(metric_configs):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    if key is not None:
        dv = np.mean(dqn_data[key])
        pv = np.mean(ppo_data[key])
    else:
        dv = np.mean(dqn_data["high_r"])
        pv = np.mean(ppo_data["high_r"])

    bars = ax.bar(["DQN", "PPO"], [dv, pv], color=[DQN_COLOR, PPO_COLOR],
                  width=0.45, edgecolor="white")
    for bar, val in zip(bars, [dv, pv]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                fmt.format(val), ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel("Value")

fig.suptitle("Performance Summary Dashboard — DQN vs PPO\nDynamic Spectrum Allocation under QoS-Aware Stochastic Traffic",
             fontweight="bold")
plt.savefig("results/publication/plot7_summary_dashboard.png", dpi=300, bbox_inches="tight")
print("Saved: plot7_summary_dashboard.png")
plt.close()

print("\n✅  All 7 publication plots saved to results/publication/")