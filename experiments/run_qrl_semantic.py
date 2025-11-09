import torch
import numpy as np
import yaml
import argparse
import os
import sys

# Thêm đường dẫn thư mục gốc vào sys.path
# Điều này giúp import các module từ thư mục 'src' một cách nhất quán
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environments.uav_swarm_env import UAVSwarmEnv
from src.core.replay_buffer import ReplayBuffer
from src.core.trainer import Trainer
from src.agents.q_agent import QRLDDPGAgent # <<<<<<< Agent QRL

def set_seed(seed):
    """Đặt seed cho reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True # Có thể làm chậm, bật nếu cần
    # torch.backends.cudnn.benchmark = False

def main(config):
    # --- Khởi tạo Môi trường ---
    env = UAVSwarmEnv(
        num_uavs=config['num_uavs'],
        num_targets=config['num_targets'],
        area_size=config['area_size']
    )

    # Đặt seed
    set_seed(config['seed'])

    # --- Lấy thông tin về kích thước ---
    obs_dim = env.observation_space.shape[1]
    action_dim_move = env.action_space[0][0].n
    action_dim_comm = env.action_space[0][1].n


    # Kích thước cho Centralized Critic
    global_obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    # Hành động chung là tổng số hành động của các agent
    # (Mỗi agent có 2 hành động, mỗi hành động là one-hot)
    global_action_dim = env.num_uavs * (action_dim_move + action_dim_comm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # --- Khởi tạo các Agents ---
    agents = []
    for i in range(config['num_uavs']):
        agent = QRLDDPGAgent(
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
            device=device,
            n_qubits=config['n_qubits'],
            n_layers=config['n_layers']
        )
        agents.append(agent)

    # --- Khởi tạo Replay Buffer ---
    # Kích thước action trong buffer là số lượng hành động rời rạc (chưa one-hot)
    buffer_action_dim = config['num_uavs'] * 2
    replay_buffer = ReplayBuffer(global_obs_dim, buffer_action_dim)

    # --- Khởi tạo Trainer ---
    trainer = Trainer(env, agents, replay_buffer, config)

    # --- Bắt đầu Huấn luyện ---
    print("Bắt đầu quá trình huấn luyện Agent QRL...")
    trainer.train()

    # --- Lưu model cuối cùng ---
    final_model_path = os.path.join(config['results_path'], "saved_models", "qrl_final")
    print(f"Huấn luyện hoàn tất. Đang lưu model cuối cùng vào {final_model_path}")
    trainer.save_models(final_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn đến file config")
    args = parser.parse_args()

    # Đọc file config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)

