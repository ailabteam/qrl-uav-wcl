import torch
import numpy as np
import yaml
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environments.uav_swarm_env import UAVSwarmEnv
from src.core.replay_buffer import ReplayBuffer
from src.core.trainer import Trainer
from src.agents.d_agent import DDPGAgent # <<<<<<< Agent DRL

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main(config):
    env = UAVSwarmEnv(
        num_uavs=config['num_uavs'],
        num_targets=config['num_targets'],
        area_size=config['area_size']
    )

    set_seed(config['seed'])

    obs_dim = env.observation_space.shape[1]
    action_dim_move = env.action_space[0][0].n
    action_dim_comm = env.action_space[0][1].n

    global_obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    global_action_dim = env.num_uavs * (action_dim_move + action_dim_comm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    agents = []
    for i in range(config['num_uavs']):
        agent = DDPGAgent(
            agent_id=i,
            obs_dim=obs_dim,
            action_dim_move=action_dim_move,
            action_dim_comm=action_dim_comm,
            global_obs_dim=global_obs_dim,
            global_action_dim=global_action_dim,
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            gamma=config['gamma'],
            tau=config['tau'],
            device=device
        )
        agents.append(agent)

    buffer_action_dim = config['num_uavs'] * 2
    replay_buffer = ReplayBuffer(global_obs_dim, buffer_action_dim)

    trainer = Trainer(env, agents, replay_buffer, config)

    print("Bắt đầu quá trình huấn luyện Agent DRL (Baseline)...")
    trainer.train()

    final_model_path = os.path.join(config['results_path'], "saved_models", "drl_final")
    print(f"Huấn luyện hoàn tất. Đang lưu model cuối cùng vào {final_model_path}")
    trainer.save_models(final_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn đến file config")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)

