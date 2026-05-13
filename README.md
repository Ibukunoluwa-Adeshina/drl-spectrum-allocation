# Deep Reinforcement Learning for Dynamic Spectrum Allocation

## Overview

This project implements a Deep Reinforcement Learning (DQN)-based dynamic spectrum allocation framework for wireless communication systems.

The system simulates multiple wireless users competing for limited spectrum resources while an RL agent learns intelligent channel allocation strategies that maximize throughput and minimize interference.

---

## Features

- Custom wireless communication environment
- DQN-based spectrum allocation
- SINR-aware reward formulation
- Throughput optimization
- Interference-aware learning
- Reward logging and visualization
- Research-oriented project structure

---

## Technologies Used

- Python
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy
- Matplotlib

---

## Project Structure

```plaintext
drl-spectrum-allocation/
│
├── env/
├── train/
├── plots/
├── results/
├── paper/
├── README.md
└── requirements.txt
```

---

## Reward Formulation

The reinforcement learning agent optimizes wireless spectrum usage using a reward function based on throughput and SINR:

\[
Reward = (\text{Throughput} \times 5) + (\text{SINR} \times 2) - \text{Congestion Penalty}
\]

---

## Current Capabilities

- Dynamic channel allocation
- Interference-aware optimization
- Reward tracking
- Learning curve visualization
- DQN training and evaluation

---

## Future Improvements

- Multi-Agent Reinforcement Learning
- OFDMA Resource Allocation
- Massive MIMO Support
- Mobility Modeling
- Federated Reinforcement Learning
- QoS-aware Scheduling

---

## Author

Ibukunoluwa Sunday Adeshina

B.Eng Telecommunications Engineering  
Federal University of Technology, Minna  
Research Interests: Wireless Networks, AI/ML, 6G Systems, Intelligent Spectrum Management