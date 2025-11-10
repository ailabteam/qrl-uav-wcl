import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import csv  # <<< THÊM DÒNG NÀY
from tqdm import tqdm

class Trainer:
    """
    Lớp Trainer điều khiển vòng lặp huấn luyện cho các agent.
    Sử dụng kiến trúc Centralized Training with Decentralized Execution (CTDE).
    """
    def __init__(self, env, agents, replay_buffer, config, agent_type='unknown'):
        self.env = env
        self.agents = agents # Danh sách các agent
        self.num_agents = len(agents)
        self.replay_buffer = replay_buffer
        self.config = config # File config chứa hyperparameters

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lấy thông tin từ môi trường
        # Giả định tất cả agent có cùng obs/action space
        self.obs_dim = env.observation_space.shape[1]
        self.action_dim_move = env.action_space[0][0].n
        self.action_dim_comm = env.action_space[0][1].n
        self.total_action_dim = self.action_dim_move + self.action_dim_comm

        # Hyperparameters
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.max_timesteps = config['max_timesteps']
        self.start_timesteps = config['start_timesteps']
        self.max_episode_steps = config['max_episode_steps'] # <<<< THÊM DÒNG NÀY


        # --- THÊM KHỐI CODE SAU VÀO CUỐI HÀM __init__ ---
        self.agent_type = agent_type
        self.results_path = config.get('results_path', 'results') # Dùng .get để an toàn
        log_dir = os.path.join(self.results_path, "logs")
        os.makedirs(log_dir, exist_ok=True) # Tạo thư mục nếu chưa có

        self.log_filepath = os.path.join(log_dir, f"{self.agent_type}_log.csv")
        print(f"Dữ liệu log sẽ được lưu tại: {self.log_filepath}")

        # Tạo file log và viết header (tiêu đề cột)
        with open(self.log_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Total Timesteps', 'Episode Reward', 'Episode Length'])
        # ---------------------------------------------------



    def train(self):
        """
        Vòng lặp huấn luyện chính. (PHIÊN BẢN NÂNG CẤP VỚI LOGGING)
        """
        obs, info = self.env.reset()
        episode_reward = 0
        episode_timesteps = 0
        
        # Biến đếm số lần cập nhật mạng
        num_updates = 0
        
        # Sử dụng tqdm để tạo thanh tiến trình
        print("Bắt đầu vòng lặp huấn luyện chính...")
        progress_bar = tqdm(range(int(self.max_timesteps)), desc="Tổng tiến trình")

        for t in progress_bar: # Thay vòng lặp for cũ bằng tqdm
            episode_timesteps += 1
            
            # 1. Chọn hành động
            if t < self.start_timesteps:
                actions_tuple = self.env.action_space.sample()
            else:
                actions_list = []
                for i in range(self.num_agents):
                    agent_obs = torch.FloatTensor(obs[i]).to(self.device)
                    move_act, comm_act = self.agents[i].select_action(agent_obs)
                    actions_list.append((move_act.cpu().item(), comm_act.cpu().item()))
                actions_tuple = tuple(actions_list)

            # 2. Thực thi hành động trong môi trường
            next_obs, reward, terminated, truncated, info = self.env.step(actions_tuple)
            done = terminated or truncated

            # 3. Lưu kinh nghiệm vào Replay Buffer
            actions_to_save = np.array(actions_tuple).flatten()
            flat_obs = obs.flatten()
            flat_next_obs = next_obs.flatten()
            self.replay_buffer.add(flat_obs, actions_to_save, reward, flat_next_obs, float(done))

            obs = next_obs
            episode_reward += reward

            # --- LOGGING CHI TIẾT GIAI ĐOẠN ĐẦU ---
            if t == self.start_timesteps:
                print(f"\n[INFO] Đã hoàn thành {self.start_timesteps} bước khởi tạo. Bắt đầu cập nhật mạng...")
            # ----------------------------------------
            
            # 4. Huấn luyện các agent
            if t >= self.start_timesteps:
                for agent in self.agents:
                    agent.update(self.replay_buffer, self.batch_size, self.agents)
                num_updates += 1

            # --- LOGGING TIẾN TRÌNH CẬP NHẬT ---
            if t >= self.start_timesteps and t % 1000 == 0: # In ra mỗi 1000 bước
                progress_bar.set_postfix({
                    "Episode Reward": f"{episode_reward:.2f}",
                    "Num Updates": num_updates
                })
            # ------------------------------------
                
            # Giả định một episode có độ dài tối đa để reset
            # Điều này rất quan trọng, nếu không chương trình sẽ chạy mãi trong một episode
            max_episode_steps = 250 # <<<<<< THÊM GIỚI HẠN NÀY
            if done or (episode_timesteps >= max_episode_steps):
                # Ghi log ra file CSV
                with open(self.log_filepath, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([t+1, episode_reward, episode_timesteps])
                
                # In ra terminal
                print(f"\n--- Episode End ---")
                print(f"Total T: {t+1}/{int(self.max_timesteps)}   Episode Reward: {episode_reward:.3f}   Episode T: {episode_timesteps}")
                print(f"-------------------")

                obs, info = self.env.reset()
                episode_reward = 0
                episode_timesteps = 0


    def save_models(self, path):
        """Lưu model của tất cả các agent."""
        if not os.path.exists(path):
            os.makedirs(path)
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(path, f"agent_{i}"))

    def load_models(self, path):
        """Load model cho tất cả các agent."""
        for i, agent in enumerate(self.agents):
            agent.load(os.path.join(path, f"agent_{i}"))

