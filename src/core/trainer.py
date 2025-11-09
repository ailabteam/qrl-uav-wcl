import torch
import torch.nn.functional as F
import numpy as np
import os
import time

class Trainer:
    """
    Lớp Trainer điều khiển vòng lặp huấn luyện cho các agent.
    Sử dụng kiến trúc Centralized Training with Decentralized Execution (CTDE).
    """
    def __init__(self, env, agents, replay_buffer, config):
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

    def train(self):
        """
Vòng lặp huấn luyện chính.
        """
        obs, info = self.env.reset()
        episode_reward = 0
        episode_timesteps = 0

        start_time = time.time()

        for t in range(int(self.max_timesteps)):
            episode_timesteps += 1

            # --- TÌM VÀ THAY THẾ KHỐI CODE TỪ "1. Chọn hành động" ĐẾN HẾT DÒNG "self.replay_buffer.add..." ---

            # 1. Chọn hành động
            if t < self.start_timesteps:
                # Hành động ngẫu nhiên, sample() sẽ trả về đúng định dạng tuple lồng nhau
                actions_tuple = self.env.action_space.sample()
            else:
                # Agent chọn hành động và tạo ra tuple lồng nhau
                actions_list = []
                for i in range(self.num_agents):
                    agent_obs = torch.FloatTensor(obs[i]).to(self.device) # obs từ env luôn ở trên CPU
                    # --- THAY ĐỔI Ở ĐÂY ---
                    # Lấy hành động (là tensor trên GPU) và chuyển nó về CPU, sau đó lấy giá trị số
                    move_act, comm_act = self.agents[i].select_action(agent_obs)
                    actions_list.append((move_act.cpu().item(), comm_act.cpu().item()))
                actions_tuple = tuple(actions_list)

            # 2. Thực thi hành động trong môi trường
            # env.step() giờ đây nhận vào đúng định dạng nó mong đợi
            next_obs, reward, terminated, truncated, info = self.env.step(actions_tuple)
            done = terminated or truncated

            # 3. Lưu kinh nghiệm vào Replay Buffer
            # --- THAY ĐỔI Ở ĐÂY ---
            # actions_tuple giờ đã chứa các số Python thông thường, nên np.array sẽ hoạt động
            actions_to_save = np.array(actions_tuple).flatten()

            # Chúng ta lưu trạng thái và hành động chung cho centralized critic
            flat_obs = obs.flatten()
            flat_next_obs = next_obs.flatten()
            self.replay_buffer.add(flat_obs, actions_to_save, reward, flat_next_obs, float(done))


            obs = next_obs
            episode_reward += reward

            # 4. Huấn luyện các agent
            if t >= self.start_timesteps:
                for agent in self.agents:
                    # Mỗi agent có thể có logic cập nhật riêng
                    # Ở đây ta giả định tất cả dùng chung buffer và logic
                    agent.update(self.replay_buffer, self.batch_size, self.agents)


            if done:
                duration = time.time() - start_time
                print(f"Total T: {t+1}   Episode Reward: {episode_reward:.3f}   Episode T: {episode_timesteps}   Duration: {duration:.2f}s")
                obs, info = self.env.reset()
                episode_reward = 0
                episode_timesteps = 0
                start_time = time.time()

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

