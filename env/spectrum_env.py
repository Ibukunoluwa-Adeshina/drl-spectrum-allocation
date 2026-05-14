import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SpectrumEnv(gym.Env):

    def __init__(self):

        super(SpectrumEnv, self).__init__()

        # PARAMETERS
        self.num_channels = 3
        self.num_users = 5

        # QoS priorities
        self.user_priorities = {
            "high" : 2,
            "low" : 1
        }

        # action space
        self.action_space = spaces.Discrete(
            self.num_channels
        )

        # observation space
        self.observation_space = spaces.Box(
            low=0,
            high=self.num_users,
            shape=(self.num_channels,),
            dtype=np.float32
        )

        self.state = None

        # episode control
        self.current_step = 0
        self.max_steps = 50

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.state = np.zeros(
            self.num_channels,
            dtype=np.float32
        )

        self.current_step = 0

        return self.state, {}

    def step(self, action):

        self.current_step += 1

        # reset occupancy
        self.state = np.zeros(
            self.num_channels,
            dtype=np.float32
        )

        # dynamic traffic load
        active_users = np.random.randint(
            1,
            self.num_users
        )
        # active users choose channels
        other_users = np.random.randint(
            0,
            self.num_channels,
            size=active_users
        )

        # randomly assign traffic type
        traffic_type = np.random.choice(
            ["high", "low"]
        )
        priority_weight = self.user_priorities[
            traffic_type
        ]

        # occupancy count
        for user_channel in other_users:
            self.state[user_channel] += 1

        # RL agent allocation
        self.state[action] += 1

        occupancy = self.state[action]

        # throughput
        throughput = 1 / occupancy

        # SINR
        signal_power = 1.0
        interference_power = occupancy - 1
        noise = 0.1

        sinr = signal_power / (
            interference_power + noise
        )

        # reward
        reward = (
            throughput * 5
        ) + (
            sinr * 2
        ) * priority_weight

        if occupancy >= 3:
            reward -= 10

        # episode termination
        done = self.current_step >= self.max_steps


        
        return (
            self.state,
            reward,
            done,
            False,
            {
                "throughput": throughput,
                "sinr": sinr,
                "occupancy": occupancy,
                "traffic_type": traffic_type,
                "priority_weight": priority_weight
            }
        )