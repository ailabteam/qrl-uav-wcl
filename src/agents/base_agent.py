from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Lớp cơ sở trừu tượng cho các agent Reinforcement Learning.
    """
    def __init__(self, agent_id, obs_dim, action_dim):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    @abstractmethod
    def select_action(self, obs, explore=True):
        """
        Chọn một hành động dựa trên quan sát.
        :param obs: Quan sát của agent.
        :param explore: True nếu đang trong giai đoạn khám phá, False nếu là giai đoạn đánh giá.
        """
        pass

    @abstractmethod  # <<<<<<< ĐÃ SỬA LỖI Ở ĐÂY
    def update(self, replay_buffer, batch_size, agents):

        """
        Cập nhật các mạng của agent từ một batch dữ liệu.
        :param replay_buffer: Buffer chứa các kinh nghiệm.
        :param batch_size: Kích thước của batch để lấy mẫu.
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Lưu trạng thái của agent (trọng số mạng).
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        Tải trạng thái của agent.
        """
        pass

