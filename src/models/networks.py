import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalActor(nn.Module):
    """
    Một mạng Actor cổ điển (MLP) cho hành động rời rạc.
    Nó sẽ output ra logits cho mỗi hành động.
    """
    def __init__(self, state_dim, action_dim_move, action_dim_comm):
        super().__init__()
        self.action_dim_move = action_dim_move
        self.action_dim_comm = action_dim_comm

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Hai "đầu" riêng biệt cho hai loại hành động
        self.move_head = nn.Linear(128, action_dim_move)
        self.comm_head = nn.Linear(128, action_dim_comm)

    def forward(self, state):
        x = self.net(state)
        move_logits = self.move_head(x)
        comm_logits = self.comm_head(x)
        return move_logits, comm_logits

    def get_action(self, state, deterministic=False):
        """Lấy hành động từ state."""
        move_logits, comm_logits = self.forward(state)
        
        # Sử dụng Gumbel-Softmax trick cho exploration hoặc argmax cho exploitation
        if deterministic:
            move_action = torch.argmax(move_logits, dim=-1)
            comm_action = torch.argmax(comm_logits, dim=-1)
        else:
            move_dist = torch.distributions.Categorical(logits=move_logits)
            move_action = move_dist.sample()
            comm_dist = torch.distributions.Categorical(logits=comm_logits)
            comm_action = comm_dist.sample()
            
        return move_action, comm_action

class ClassicalCritic(nn.Module):
    """
    Một mạng Critic cổ điển (MLP).
    Input: state và action. Output: Q-value.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

if __name__ == '__main__':
    # Đoạn code kiểm tra nhanh
    print("--- Chạy kiểm tra các mô hình cổ điển ---")
    state_dim = 10
    action_dim_move = 5
    action_dim_comm = 3
    total_action_dim = action_dim_move + action_dim_comm # Để test Critic

    batch_size = 4
    dummy_state = torch.randn(batch_size, state_dim)
    dummy_action = torch.randn(batch_size, total_action_dim)

    # Test Actor
    actor = ClassicalActor(state_dim, action_dim_move, action_dim_comm)
    move_logits, comm_logits = actor(dummy_state)
    print(f"Kích thước đầu ra của Actor (move): {move_logits.shape}")
    print(f"Kích thước đầu ra của Actor (comm): {comm_logits.shape}")
    assert move_logits.shape == (batch_size, action_dim_move)
    assert comm_logits.shape == (batch_size, action_dim_comm)
    
    move_act, comm_act = actor.get_action(dummy_state)
    print(f"Hành động được chọn (move): {move_act.shape}")
    print(f"Hành động được chọn (comm): {comm_act.shape}")

    # Test Critic
    critic = ClassicalCritic(state_dim, total_action_dim)
    q_values = critic(dummy_state, dummy_action)
    print(f"\nKích thước đầu ra của Critic: {q_values.shape}")
    assert q_values.shape == (batch_size, 1)

    print("--- Kiểm tra hoàn tất ---")
