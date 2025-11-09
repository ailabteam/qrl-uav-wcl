# Quantum-Accelerated Semantic Control for Task-Oriented UAV Swarms

## Overview

This repository contains the source code for the paper submitted to *IEEE Wireless Communications Letters (WCL)*. The project investigates the application of Quantum Reinforcement Learning (QRL) to solve a semantic-aware control problem for a swarm of UAVs performing a task-oriented mission.

The primary goal is to demonstrate that a QRL-based agent can learn complex, cooperative strategies more efficiently and achieve higher task performance compared to classical Deep Reinforcement Learning (DRL) baselines.

## Project Structure

```
qrl-uav-wcl/
├── configs/               # Hyperparameter configuration files
├── experiments/           # Scripts to run experiments
├── notebooks/             # Jupyter notebooks for analysis and plotting
├── results/               # Directory to save logs, figures, and models
├── src/                   # Main source code
│   ├── agents/            # RL agent implementations (QRL, DRL)
│   ├── core/              # Core components like trainer and replay buffer
│   ├── environments/      # Custom UAV swarm environment
│   └── models/            # Neural network and quantum circuit architectures
├── .gitignore
└── README.md
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-link]
    cd qrl-uav-wcl
    ```

2.  **Create and activate Conda environment:**
    ```bash
    conda create -n qrl_uav python=3.11 -y
    conda activate qrl_uav
    ```

3.  **Install dependencies:**
    ```bash
    # Install PyTorch for CUDA 12.1+
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # Install other libraries
    pip install pennylane pennylane-lightning[gpu] gymnasium matplotlib pandas tqdm scikit-learn
    ```

## How to Run

To run an experiment, use the scripts in the `experiments` directory. For example:

```bash
# To run the QRL agent
python experiments/run_qrl_semantic.py --config configs/config.yaml

# To run the DRL baseline
python experiments/run_drl_baseline.py --config configs/config.yaml
```

## Results

Figures and plots will be saved in the `results/figures/` directory with a resolution of 600 DPI.
