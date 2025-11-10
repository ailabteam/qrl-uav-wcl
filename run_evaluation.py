# file: run_evaluation.py

import torch
import numpy as np
import yaml
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm

# Thêm 'src' vào path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environments.uav_swarm_env import UAVSwarmEnv
from src.agents.q_agent import QRLDDPGAgent
from src.agents.d_agent import DDPGAgent

def evaluate_agent(config, agent_type, model_path, num_episodes=100):
    """
    Hàm để đánh giá một agent đã được huấn luyện.
    """
    print(f"\n--- Bắt đầu Đánh giá cho Agent: {agent_type.upper()} ---")
    
    # --- 1. Khởi tạo Môi trường ---
    # Chạy ở chế độ 'rgb_array' để có thể lưu video nếu cần, và không hiện cửa sổ pygame
    env = UAVSwarmEnv(
        num_uavs=config['num_uavs'],
        num_targets=config['num_targets'],
        area_size=config['area_size'],
        render_mode=None 
    )
    
    # --- 2. Khởi tạo Agent ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Lấy thông tin về kích thước
    obs_dim = env.observation_space.shape[1]
    action_dim_move = env.action_space[0][0].n
    action_dim_comm = env.action_space[0][1].n
    global_obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    global_action_dim = env.num_uavs * (action_dim_move + action_dim_comm)

    agents = []
    agent_class = QRLDDPGAgent if agent_type == 'qrl' else DDPGAgent
    
    for i in range(config['num_uavs']):
        agent_params = {
            'agent_id': i, 'obs_dim': obs_dim, 'action_dim_move': action_dim_move,
            'action_dim_comm': action_dim_comm, 'global_obs_dim': global_obs_dim,
            'global_action_dim': global_action_dim, 'actor_lr': config['actor_lr'],
            'critic_lr': config['critic_lr'], 'gamma': config['gamma'], 'tau': config['tau'],
            'device': device
        }
        if agent_type == 'qrl':
            agent_params.update({'n_qubits': config['n_qubits'], 'n_layers': config['n_layers']})
        
        agent = agent_class(**agent_params)
        
        # Tải trọng số đã huấn luyện
        try:
            agent_model_path = os.path.join(model_path, f"agent_{i}")
            agent.load(agent_model_path)
            print(f"   - Tải thành công model cho agent {i} từ {agent_model_path}")
        except FileNotFoundError:
            print(f"[LỖI] Không tìm thấy file model tại {agent_model_path}. Dừng lại.")
            return None
            
        agents.append(agent)

    # --- 3. Chạy Vòng lặp Đánh giá ---
    all_episode_stats = []
    
    for episode in tqdm(range(num_episodes), desc=f"Đánh giá {agent_type.upper()}"):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        # Reset lại các chỉ số nhiệm vụ
        confirmed_tois = 0
        total_tois = len(env.classified_targets)

        while not done and episode_steps < config['max_episode_steps']:
            actions_list = []
            for i in range(config['num_uavs']):
                # Chạy với deterministic=True để có kết quả tốt nhất
                move_act, comm_act = agents[i].select_action(obs[i], explore=False)
                actions_list.append((move_act.item(), comm_act.item()))
            actions_tuple = tuple(actions_list)

            obs, reward, terminated, truncated, info = env.step(actions_tuple)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

        # Thu thập kết quả cuối cùng của episode
        # Đếm số ToI đã được xác nhận
        for toi_status in env.classified_targets.values():
            if toi_status['confirmed']:
                confirmed_tois += 1
        
        confirmation_rate = (confirmed_tois / total_tois) if total_tois > 0 else 0
        
        all_episode_stats.append({
            'Episode': episode,
            'Total Reward': episode_reward,
            'Confirmation Rate': confirmation_rate * 100, # Chuyển sang %
            'Episode Length': episode_steps
        })

    # --- 4. Tổng hợp và Trả về Kết quả ---
    df_stats = pd.DataFrame(all_episode_stats)
    
    summary = {
        'Agent': agent_type.upper(),
        'Mean Reward': df_stats['Total Reward'].mean(),
        'Mean Confirmation Rate (%)': df_stats['Confirmation Rate'].mean(),
        'Std Confirmation Rate (%)': df_stats['Confirmation Rate'].std()
    }
    
    print(f"--- Kết quả Đánh giá cho {agent_type.upper()} ---")
    print(f"  - Phần thưởng Trung bình: {summary['Mean Reward']:.2f}")
    print(f"  - Tỷ lệ Xác nhận ToI Trung bình: {summary['Mean Confirmation Rate (%)']:.2f}%")
    print("-" * 30)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn đến file config")
    parser.add_argument("--episodes", type=int, default=100, help="Số lượng episode để đánh giá")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    results_summary = []
    
    # Đánh giá DRL
    drl_model_path = os.path.join(config['results_path'], "saved_models", "drl_final")
    drl_summary = evaluate_agent(config, 'drl', drl_model_path, args.episodes)
    if drl_summary:
        results_summary.append(drl_summary)

    # Đánh giá QRL
    qrl_model_path = os.path.join(config['results_path'], "saved_models", "qrl_final")
    qrl_summary = evaluate_agent(config, 'qrl', qrl_model_path, args.episodes)
    if qrl_summary:
        results_summary.append(qrl_summary)
        
    # Lưu kết quả tổng hợp vào file CSV để chuẩn bị vẽ biểu đồ
    if results_summary:
        df_final = pd.DataFrame(results_summary)
        output_path = os.path.join(config['results_path'], "evaluation_summary.csv")
        df_final.to_csv(output_path, index=False)
        print(f"\nKết quả tổng hợp đã được lưu tại: {output_path}")
        print(df_final.to_string())
