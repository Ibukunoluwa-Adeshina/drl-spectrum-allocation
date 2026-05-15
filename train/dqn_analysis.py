import numpy as np
from stable_baselines3 import DQN
from env.spectrum_env import SpectrumEnv


def main():

    env = SpectrumEnv()

    model = DQN.load(
        "results/dqn_spectrum_model"
    )

    NUM_EPISODES = 100

    reward_list = []
    throughput_list = []
    sinr_list = []

    for episode in range(NUM_EPISODES):

        obs, _ = env.reset()

        done = False

        total_reward = 0

        while not done:

            action, _ = model.predict(obs)

            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward

            throughput_list.append(
                info["throughput"]
            )

            sinr_list.append(
                info["sinr"]
            )

        reward_list.append(total_reward)

    print("\n===== DQN PERFORMANCE =====")

    print(
        f"Average Reward: "
        f"{np.mean(reward_list):.2f}"
    )

    print(
        f"Average Throughput: "
        f"{np.mean(throughput_list):.4f}"
    )

    print(
        f"Average SINR: "
        f"{np.mean(sinr_list):.4f}"
    )


if __name__ == "__main__":
    main()