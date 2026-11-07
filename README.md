# 🚗 Adaptive Cruise Control Using Deep Reinforcement Learning

### PPO-Based Autonomous Driving in Dynamic Highway Environments

An academic Deep Reinforcement Learning project exploring intelligent Adaptive Cruise Control (ACC) through **Proximal Policy Optimization (PPO)** in simulated highway environments.

The project investigates how autonomous driving agents can learn safe, efficient, and adaptive driving policies through interaction with dynamic traffic environments, using reward optimization and Actor-Critic neural network architectures.

Developed as part of the Reinforcement Learning course at the **National Higher School of Artificial Intelligence (ENSIA)**, May 2025.

---

## 🎯 Project Overview

Adaptive Cruise Control (ACC) is an essential component of Advanced Driver Assistance Systems (ADAS), allowing vehicles to automatically regulate their speed and maintain safe distances from surrounding traffic.

Traditional rule-based and classical control approaches can struggle with complex traffic situations, unpredictable driver behavior, and dynamic highway conditions.

This project explores Deep Reinforcement Learning as an alternative approach.

We formulate autonomous highway driving as a **Markov Decision Process (MDP)** and train an intelligent agent using Proximal Policy Optimization (PPO).

Through repeated interactions with the simulated environment, the agent learns to:

- Maintain appropriate driving speeds.
- Avoid collisions with surrounding vehicles.
- Maintain safe following distances.
- Perform strategic lane changes.
- Overtake slower vehicles.
- Adapt driving decisions to changing traffic conditions.

---

## 🏗️ System Architecture

The project follows a reinforcement learning pipeline composed of environment simulation, state observation, policy inference, reward evaluation, and iterative model optimization.

### 1. Highway Environment Simulation

The driving environment is implemented using [Highway-Env](https://github.com/Farama-Foundation/HighwayEnv), a simulation framework for autonomous highway driving.

The environment incorporates:

- Multi-lane highway configurations.
- Dynamic surrounding vehicles.
- Aggressive driving behavior models.
- Randomized vehicle positions and velocities.
- Collision detection.
- Lane-changing dynamics.

**Experimental configuration documented in the research report:**

| Parameter | Value |
|---|---|
| Number of lanes | 4 |
| Surrounding vehicles | 50 |
| Episode duration | 60 seconds |
| Policy frequency | 15 |
| Controlled vehicles | 1 |
| Other vehicle behavior | AggressiveVehicle |
| Target speed range | 20–30 m/s |

### 2. Markov Decision Process Formulation

The Adaptive Cruise Control problem is formulated as an MDP:

**State Space**

The agent observes information about its driving environment, including:

- Ego vehicle speed and acceleration.
- Relative positions of surrounding vehicles.
- Distances to nearby vehicles.
- Current lane index.
- Adjacent lane availability.

**Action Space**

The agent can select from five discrete driving actions:

| Action | Description |
|---|---|
| LANE_LEFT | Change to the left lane |
| IDLE | Maintain current behavior |
| LANE_RIGHT | Change to the right lane |
| FASTER | Accelerate |
| SLOWER | Decelerate |

**Reward Function**

A reward-shaping strategy encourages safe and efficient driving.

The reward components include:

- Collision penalties.
- Positive rewards for maintaining target speeds.
- Penalties for unnecessary lane changes.
- Rewards for efficient driving behavior.
- Additional incentives for safe overtaking.

### 3. Proximal Policy Optimization (PPO)

The primary reinforcement learning algorithm is Proximal Policy Optimization.

PPO is a policy-gradient algorithm that uses a clipped surrogate objective to constrain policy updates and improve training stability.

The implementation explores the Actor-Critic architecture.

**Actor Network**

Learns the driving policy and estimates action probabilities based on the current state.

**Critic Network**

Estimates the value of the current state and provides feedback to guide policy optimization.

The custom PyTorch implementation includes:

- Separate Actor and Critic neural networks.
- Two hidden layers with 128 neurons each.
- SiLU activation functions.
- Softmax action probability estimation.
- Action masking for unavailable actions.
- Clipped policy optimization.
- Importance sampling ratios.
- Temporal-difference error.
- Separate Adam optimizers for Actor and Critic.

An additional PPO implementation using Stable-Baselines3 is also included.

### 4. Training and Optimization

The training pipeline allows the agent to interact with the highway simulator and progressively improve its policy.

The workflow includes:

1. Environment initialization.
2. State observation.
3. Action selection.
4. Environment interaction.
5. Reward collection.
6. Policy and value network optimization.
7. Model checkpointing.
8. Evaluation in simulated traffic scenarios.

The implementation includes GPU acceleration support through PyTorch.

---

## 📊 Experimental Results

According to the experimental report, the PPO agent was trained for 100,000 timesteps.

### Learning Performance

The training experiments demonstrated progressive improvement in the agent's cumulative rewards.

| Training Timesteps | Approximate Average Reward |
|---|---:|
| 10,000 | 75 |
| 100,000 | 195 |

The report also describes increasing episode lengths during training, suggesting improved driving stability and fewer premature episode terminations.

### Learned Driving Behaviors

During evaluation, the trained agent demonstrated:

- Adaptive speed regulation.
- Collision avoidance.
- Safe-distance maintenance.
- Strategic lane changes.
- Overtaking slower vehicles.
- Improved stability in dense traffic.

### Algorithm Comparison

The research also investigated alternative reinforcement learning algorithms.

| Algorithm | Reported Observations |
|---|---|
| PPO | Best overall convergence and learning stability |
| SAC | Competitive rewards, with less stable lane selection |
| DDPG | Sensitivity to reward shaping and generalization challenges |
| DQN | Lower performance in the evaluated scenarios |

These comparisons reflect the experiments described in the accompanying academic report.

---

## 🧪 Evaluation Methodology

The trained agent was evaluated using several performance indicators:

**1. Average Episode Reward**

Measures the cumulative reward obtained during an episode.

**2. Collision Rate**

Measures the frequency of collision-related episode terminations.

**3. Episode Duration**

Evaluates how long the agent can operate before an episode terminates.

**4. Lane Change Efficiency**

Assesses whether the agent makes appropriate lane-changing decisions.

**5. Speed Maintenance**

Measures the agent's ability to maintain the desired speed range.

The project also includes evaluation scripts for running trained models across multiple episodes and generating reward visualizations.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Reinforcement Learning | Proximal Policy Optimization |
| RL Framework | Stable-Baselines3 |
| Simulation Environment | Highway-Env |
| Environment Interface | Gymnasium / Gym |
| Neural Network Architecture | Actor-Critic, MLP |
| Optimization | Adam |
| Numerical Computing | NumPy |
| Data Visualization | Matplotlib |
| Configuration | YAML |

---

## 📁 Repository Structure

```text
adaptive-cruise-control-ppo/
│
├── main.py
├── manual_control.py
├── test_agent_PPO.py
│
├── training/
│   ├── agent_PPO.py
│   ├── train.py
│   └── config.yaml
│
├── Models/
│   ├── acc_agent.py
│   ├── base_agent.py
│   ├── utils.py
│   ├── hyper.py
│   ├── your_baseline.py
│   │
│   └── PPO/
│       ├── PPO_actor_model_25000.pt
│       ├── PPO_actor_model_50000.pt
│       ├── PPO_actor_model_75000.pt
│       ├── PPO_actor_model_100000.pt
│       └── ...
│
├── Figures/
│
├── requirements.txt
├── .gitignore
└── README.md
```

The repository includes training scripts, PPO implementations, environment utilities, evaluation scripts, and saved model checkpoints.

---

## 🚀 Installation

### Requirements

- Python 3.10+
- PyTorch
- Gymnasium
- Highway-Env
- Stable-Baselines3
- NumPy
- Matplotlib
- PyYAML

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-cruise-control-ppo.git

cd adaptive-cruise-control-ppo
```

### 2. Create a virtual environment

```bash
python -m venv venv

source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch numpy matplotlib pyyaml
pip install gymnasium highway-env stable-baselines3
```

### 4. Run the simulation

The repository contains training and evaluation scripts.

Before running, ensure the environment configuration, Python import paths, and selected PPO implementation are consistent with your installed dependency versions.

**Note:** The archived implementation is an academic experimental codebase. Some scripts require integration and compatibility adjustments before end-to-end reproduction.

---

## 🔬 Research Contributions

The main technical contributions of this project include:

- Formulating adaptive cruise control as a Markov Decision Process.
- Implementing PPO-based autonomous driving policies.
- Developing custom Actor-Critic neural networks using PyTorch.
- Exploring clipped policy optimization and action masking.
- Designing reward-shaping strategies for safety and driving efficiency.
- Training and evaluating agents in dynamic, multi-lane highway environments.
- Investigating alternative reinforcement learning algorithms.
- Analyzing learned driving behaviors through simulation and reward-based evaluation.

---

## 🚧 Limitations & Future Work

Although the experiments demonstrate promising results in simulation, several challenges remain before practical deployment.

Potential extensions include:

- Safety-constrained reinforcement learning.
- Multi-agent reinforcement learning.
- More diverse traffic scenarios.
- Robustness to sensor noise and uncertain observations.
- Hybrid model-based and reinforcement learning approaches.
- Real-time autonomous driving integration.
- Improved reward design and policy generalization.

---

## 📚 References

The project draws on research in reinforcement learning, autonomous driving, and adaptive cruise control.

Key references:

- Schulman et al. — Proximal Policy Optimization Algorithms (2017).
- Highway-Env — Open-source highway driving simulation environment.
- Stable-Baselines3 — Reliable Reinforcement Learning Implementations.
- Research on safety-aware and cooperative Adaptive Cruise Control.

---

*Exploring intelligent autonomous driving through Deep Reinforcement Learning.*
