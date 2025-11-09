import torch
import torch.nn.functional as F
import os
import numpy as np

# Thêm đường dẫn src vào sys.path để import dễ dàng
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.networks import ClassicalActor, ClassicalCritic
from .base_agent import BaseAgent

class DDPGAgent(BaseAgent):
    """
    Triển khai Agent DDPG cho một agent trong môi trường Multi-Agent (MADDPG).
    Sử dụng các thành phần hoàn toàn cổ điển.
    """
    def __init__(self, agent_id, obs_dim, action_dim_move, action_dim_comm,
                 global_obs_dim, global_action_dim,
                 actor_lr, critic_lr, gamma, tau, device):

        super().__init__(agent_id, obs_dim, action_dim_move + action_dim_comm)
        self.action_dim_move = action_dim_move
        self.action_dim_comm = action_dim_comm
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Actor Network (phi tập trung)
        self.actor = ClassicalActor(obs_dim, action_dim_move, action_dim_comm).to(device)
        self.actor_target = ClassicalActor(obs_dim, action_dim_move, action_dim_comm).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic Network (tập trung) - Sử dụng kiến trúc Twin-Critic của TD3 để ổn định
        self.critic1 = ClassicalCritic(global_obs_dim, global_action_dim).to(device)
        self.critic1_target = ClassicalCritic(global_obs_dim, global_action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)

        self.critic2 = ClassicalCritic(global_obs_dim, global_action_dim).to(device)
        self.critic2_target = ClassicalCritic(global_obs_dim, global_action_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

    def select_action(self, obs, explore=True, noise_scale=0.1):
      """
      Chọn hành động. Vì hành động là rời rạc, chúng ta sẽ sử dụng Gumbel-Softmax.
      """
      # --- KHỐI ĐÃ SỬA LỖI ---
      # Kiểm tra nếu obs chưa phải là tensor thì chuyển đổi, ngược lại thì giữ nguyên
      if not isinstance(obs, torch.Tensor):
          obs = torch.FloatTensor(obs).to(self.device)

      # Đảm bảo obs có đúng số chiều (batch_size, obs_dim)
      if obs.dim() == 1:
          obs = obs.unsqueeze(0)
      # ------------------------
      
      with torch.no_grad():
          move_logits, comm_logits = self.actor(obs)
      
      if explore:
          # Gumbel-Softmax trick for exploration in discrete action spaces
          move_action = F.gumbel_softmax(move_logits, hard=True).argmax(dim=-1)
          comm_action = F.gumbel_softmax(comm_logits, hard=True).argmax(dim=-1)
      else:
          move_action = move_logits.argmax(dim=-1)
          comm_action = comm_logits.argmax(dim=-1)
          
      # Trả về kết quả cho batch size = 1
      return move_action.squeeze(), comm_action.squeeze()

    # Mở file src/agents/d_agent.py
    # Xóa toàn bộ hàm update() cũ và thay thế bằng hàm này.

    def update(self, replay_buffer, batch_size, agents):
        """
        Cập nhật mạng Critic và Actor.
        Phiên bản đã sửa lỗi và chuẩn hóa xử lý hành động.
        """
        # Lấy mẫu từ replay buffer
        global_obs, global_actions_int, rewards, global_next_obs, dones = replay_buffer.sample(batch_size)
        
        # global_actions_int có shape (batch_size, num_agents * 2) và chứa các số nguyên
        
        # --- Cập nhật Critic ---

        with torch.no_grad():
            # Tính hành động tiếp theo từ các actor_target của TẤT CẢ agent
            next_actions_one_hot_list = []
            for i, agent in enumerate(agents):
                # Lấy quan sát cục bộ cho từng agent
                agent_next_obs = global_next_obs[:, i * agent.obs_dim : (i + 1) * agent.obs_dim]
                
                # Lấy hành động rời rạc (dạng số nguyên) từ actor target
                move_act, comm_act = agent.actor_target.get_action(agent_next_obs, deterministic=True)
                
                # Chuyển hành động rời rạc sang one-hot
                move_one_hot = F.one_hot(move_act, num_classes=self.action_dim_move).float()
                comm_one_hot = F.one_hot(comm_act, num_classes=self.action_dim_comm).float()
                
                next_actions_one_hot_list.append(move_one_hot)
                next_actions_one_hot_list.append(comm_one_hot)
            
            # Nối tất cả các hành động one-hot lại thành một vector dài
            global_next_actions_one_hot = torch.cat(next_actions_one_hot_list, dim=1)

            # Tính target Q-value
            target_Q1 = self.critic1_target(global_next_obs, global_next_actions_one_hot)
            target_Q2 = self.critic2_target(global_next_obs, global_next_actions_one_hot)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # Chuyển đổi hành động hiện tại từ buffer (dạng số nguyên) sang one-hot
        current_actions_one_hot_list = []
        for i in range(len(agents)):
            move_act_int = global_actions_int[:, i * 2].long()
            comm_act_int = global_actions_int[:, i * 2 + 1].long()
            
            move_one_hot = F.one_hot(move_act_int, num_classes=self.action_dim_move).float()
            comm_one_hot = F.one_hot(comm_act_int, num_classes=self.action_dim_comm).float()

            current_actions_one_hot_list.append(move_one_hot)
            current_actions_one_hot_list.append(comm_one_hot)
        
        global_current_actions_one_hot = torch.cat(current_actions_one_hot_list, dim=1)

        # Tính Q-value hiện tại
        current_Q1 = self.critic1(global_obs, global_current_actions_one_hot)
        current_Q2 = self.critic2(global_obs, global_current_actions_one_hot)

        # Tính Critic loss
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        # Tối ưu hóa Critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # --- Cập nhật Actor (chỉ cập nhật actor của agent hiện tại) ---
        
        # Lấy hành động dự đoán từ các actor (dạng one-hot)
        actions_pred_one_hot_list = []
        for i, agent in enumerate(agents):
            agent_obs = global_obs[:, i * agent.obs_dim : (i + 1) * agent.obs_dim]
            move_logits, comm_logits = agent.actor(agent_obs)

            # Sử dụng Gumbel-Softmax trick để giữ gradient
            move_one_hot = F.gumbel_softmax(move_logits, hard=True)
            comm_one_hot = F.gumbel_softmax(comm_logits, hard=True)

            actions_pred_one_hot_list.append(move_one_hot)
            actions_pred_one_hot_list.append(comm_one_hot)
        
        global_actions_pred_one_hot = torch.cat(actions_pred_one_hot_list, dim=1)
        
        # Tính Actor loss
        actor_loss = -self.critic1(global_obs, global_actions_pred_one_hot).mean()

        # Tối ưu hóa Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Cập nhật các mạng Target (Soft update) ---
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
        self._soft_update(self.actor_target, self.actor)

    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath):
        torch.save(self.actor.state_dict(), f"{filepath}_actor.pth")
        torch.save(self.critic1.state_dict(), f"{filepath}_critic1.pth")

    def load(self, filepath):
        self.actor.load_state_dict(torch.load(f"{filepath}_actor.pth"))
        self.critic1.load_state_dict(torch.load(f"{filepath}_critic1.pth"))

