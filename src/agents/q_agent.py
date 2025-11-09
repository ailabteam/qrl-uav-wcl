import torch
import os

# Thêm đường dẫn src vào sys.path để import dễ dàng
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.quantum_circuits import QuantumCritic
from .d_agent import DDPGAgent # Kế thừa từ DDPGAgent

class QRLDDPGAgent(DDPGAgent):
    """
    Triển khai Agent DDPG sử dụng Critic Lượng tử.
    Kế thừa từ DDPGAgent và chỉ thay thế các mạng Critic.
    """
    def __init__(self, agent_id, obs_dim, action_dim_move, action_dim_comm,
                 global_obs_dim, global_action_dim,
                 actor_lr, critic_lr, gamma, tau, device,
                 n_qubits=4, n_layers=2): # Thêm các tham số cho mạch lượng tử
        
        # Gọi __init__ của lớp cha (DDPGAgent)
        # nhưng chúng ta sẽ ghi đè các thành phần Critic ngay sau đó.
        super().__init__(agent_id, obs_dim, action_dim_move, action_dim_comm,
                         global_obs_dim, global_action_dim,
                         actor_lr, critic_lr, gamma, tau, device)

        print(f"Khởi tạo Agent {agent_id} với Quantum Critic ({n_qubits} qubits, {n_layers} layers).")

        # ----- GHI ĐÈ PHẦN CRITIC -----
        # Sử dụng QuantumCritic thay vì ClassicalCritic
        self.critic1 = QuantumCritic(global_obs_dim, global_action_dim, n_qubits, n_layers).to(device)
        self.critic1_target = QuantumCritic(global_obs_dim, global_action_dim, n_qubits, n_layers).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)

        self.critic2 = QuantumCritic(global_obs_dim, global_action_dim, n_qubits, n_layers).to(device)
        self.critic2_target = QuantumCritic(global_obs_dim, global_action_dim, n_qubits, n_layers).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Tất cả các phương thức khác (select_action, update, save, load)
        # được kế thừa trực tiếp từ DDPGAgent mà không cần thay đổi.
        # Đây chính là sức mạnh của Lập trình Hướng đối tượng!
