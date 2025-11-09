import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame # Dùng để trực quan hóa (tùy chọn nhưng rất hữu ích)

class UAVSwarmEnv(gym.Env):
    """
    Môi trường Gymnasium tùy chỉnh cho bài toán Điều khiển Ngữ nghĩa Bầy UAV.

    - Trạng thái (Observation): Vị trí của UAV, vị trí tương đối của các UAV khác,
      và thông tin cảm biến về các đối tượng mặt đất.
    - Hành động (Action): Di chuyển (5 hướng) và chính sách truyền thông (3 mức).
    - Thưởng (Reward): Dựa trên độ chính xác phân loại và độ trễ.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, num_uavs=3, num_targets=5, area_size=100, render_mode=None):
        super().__init__()

        self.num_uavs = num_uavs
        self.num_targets = num_targets
        self.area_size = area_size  # Kích thước của khu vực hoạt động (vuông)
        self.uav_speed = 5.0        # Tốc độ di chuyển của UAV
        self.target_speed = 2.0     # Tốc độ di chuyển của đối tượng
        self.uav_sensor_range = 30.0 # Tầm cảm biến của UAV
        self.communication_range = 40.0 # Tầm giao tiếp với Edge server (giả định)

        # --- THAY THẾ BẰNG KHỐI NÀY ---
        # Định nghĩa không gian hành động theo cấu trúc mới, có tổ chức hơn
        # Mỗi agent có một không gian hành động riêng là một Tuple(Di chuyển, Giao tiếp)
        agent_action_space = spaces.Tuple((
            spaces.Discrete(5, start=0), # 0: Stay, 1: Up, 2: Down, 3: Left, 4: Right
            spaces.Discrete(3, start=0)  # 0: Silent, 1: LQ, 2: HQ
        ))
        # Không gian hành động chung là một Tuple của không gian hành động của từng agent
        self.action_space = spaces.Tuple([agent_action_space] * self.num_uavs)


        # Định nghĩa không gian quan sát
        # Mỗi UAV quan sát:
        # - Vị trí của nó (2,)
        # - Vị trí tương đối của các UAV khác ( (num_uavs-1) * 2, )
        # - Vị trí tương đối của các đối tượng trong tầm cảm biến ( num_targets * 2, )
        obs_dim_per_uav = 2 + 2 * (self.num_uavs - 1) + 2 * self.num_targets
        self.observation_space = spaces.Box(
            low=-self.area_size,
            high=self.area_size,
            shape=(self.num_uavs, obs_dim_per_uav),
            dtype=np.float32
        )

        # Các biến trạng thái của môi trường
        self._uav_positions = None
        self._target_positions = None
        # Giả định một số đối tượng là "quan trọng" (ToI)
        self._target_is_toi = np.random.choice([True, False], self.num_targets, p=[0.4, 0.6])

        # Cho việc render (trực quan hóa)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 512 # Kích thước cửa sổ Pygame

    def _get_obs(self):
        """Tính toán quan sát cho tất cả các UAV."""
        all_obs = []
        for i in range(self.num_uavs):
            uav_pos = self._uav_positions[i]

            # 1. Vị trí của chính nó (đã chuẩn hóa)
            own_pos_obs = uav_pos / self.area_size

            # 2. Vị trí tương đối của UAV khác
            other_uav_obs = []
            for j in range(self.num_uavs):
                if i != j:
                    relative_pos = (self._uav_positions[j] - uav_pos) / self.area_size
                    other_uav_obs.extend(relative_pos)

            # 3. Vị trí tương đối của các đối tượng
            target_obs = []
            for k in range(self.num_targets):
                relative_pos = (self._target_positions[k] - uav_pos)
                dist = np.linalg.norm(relative_pos)
                if dist <= self.uav_sensor_range:
                    target_obs.extend(relative_pos / self.area_size)
                else:
                    # Nếu ngoài tầm, gửi 0
                    target_obs.extend([0.0, 0.0])

            obs_i = np.concatenate([own_pos_obs, np.array(other_uav_obs), np.array(target_obs)])
            all_obs.append(obs_i)
        return np.array(all_obs, dtype=np.float32)

    def _get_info(self):
        """Trả về thông tin bổ sung, hữu ích cho việc gỡ lỗi và phân tích."""
        return {
            "uav_positions": self._uav_positions,
            "target_positions": self._target_positions
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Khởi tạo vị trí ngẫu nhiên cho UAV và đối tượng
        self._uav_positions = self.np_random.uniform(0, self.area_size, size=(self.num_uavs, 2))
        self._target_positions = self.np_random.uniform(0, self.area_size, size=(self.num_targets, 2))

        # Reset trạng thái phân loại
        self.classified_targets = {k: {"confirmed": False, "confidence": 0.0} for k in range(self.num_targets) if self._target_is_toi[k]}

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, actions):
        # 1. Cập nhật vị trí UAV dựa trên hành động di chuyển
        move_actions = [agent_action[0] for agent_action in actions]

        for i, move in enumerate(move_actions):
            if move == 1: # Lên
                self._uav_positions[i][1] += self.uav_speed
            elif move == 2: # Xuống
                self._uav_positions[i][1] -= self.uav_speed
            elif move == 3: # Trái
                self._uav_positions[i][0] -= self.uav_speed
            elif move == 4: # Phải
                self._uav_positions[i][0] += self.uav_speed
            # Giữ UAV trong khu vực hoạt động
            self._uav_positions[i] = np.clip(self._uav_positions[i], 0, self.area_size)

        # 2. Cập nhật vị trí đối tượng (di chuyển ngẫu nhiên)
        random_moves = self.np_random.standard_normal(size=(self.num_targets, 2))
        self._target_positions += random_moves * self.target_speed
        self._target_positions = np.clip(self._target_positions, 0, self.area_size)

        # 3. Tính toán thưởng dựa trên hành động truyền thông
        comm_actions = [agent_action[1] for agent_action in actions]

        accuracy_gain = 0.0
        comm_cost = 0.0

        for i, comm in enumerate(comm_actions):
            if comm > 0: # Nếu UAV truyền dữ liệu
                # Giả định UAV phải ở trong tầm giao tiếp để truyền thành công
                # Đây là một mô hình đơn giản, có thể làm phức tạp hơn sau
                dist_to_gcs = np.linalg.norm(self._uav_positions[i] - [self.area_size/2, self.area_size/2]) # GCS ở giữa

                if dist_to_gcs <= self.communication_range:
                    quality = 0.5 if comm == 1 else 1.0 # LQ = 0.5, HQ = 1.0
                    comm_cost += quality # Chi phí năng lượng tỷ lệ với chất lượng

                    # Kiểm tra các đối tượng trong tầm nhìn và cập nhật độ tin cậy phân loại
                    for k in range(self.num_targets):
                        if self._target_is_toi[k] and not self.classified_targets[k]["confirmed"]:
                            dist_to_target = np.linalg.norm(self._uav_positions[i] - self._target_positions[k])
                            if dist_to_target <= self.uav_sensor_range:
                                # Độ chính xác tăng nhiều hơn khi UAV ở gần và dùng chất lượng cao
                                confidence_increase = quality * (1 - dist_to_target / self.uav_sensor_range)
                                old_confidence = self.classified_targets[k]["confidence"]
                                self.classified_targets[k]["confidence"] += confidence_increase

                                if old_confidence < 1.0 and self.classified_targets[k]["confidence"] >= 1.0:
                                    self.classified_targets[k]["confirmed"] = True
                                    accuracy_gain += 10 # Thưởng lớn khi xác nhận được một ToI mới

        # Hàm thưởng tổng hợp
        reward = accuracy_gain - 0.1 * comm_cost

        # 4. Kiểm tra điều kiện kết thúc episode (ví dụ: hết thời gian)
        # Trong trường hợp này, chúng ta giả định episode chạy vĩnh viễn
        terminated = False

        # 5. Lấy quan sát và thông tin mới
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info # Trả về truncated=False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None # render_mode="human" được xử lý trong step() và reset()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # Nền trắng
        pix_square_size = self.window_size / self.area_size

        # Vẽ đối tượng
        for i in range(self.num_targets):
            color = (255, 0, 0) if self._target_is_toi[i] else (0, 0, 0) # ToI màu đỏ, khác màu đen
            pygame.draw.circle(
                canvas,
                color,
                self._target_positions[i] * pix_square_size,
                pix_square_size * 2,
            )

        # Vẽ UAV
        for i in range(self.num_uavs):
            # Vẽ UAV
            pygame.draw.circle(
                canvas,
                (0, 0, 255), # Màu xanh
                self._uav_positions[i] * pix_square_size,
                pix_square_size * 3,
            )
            # Vẽ tầm cảm biến
            pygame.draw.circle(
                canvas,
                (0, 200, 200), # Màu xanh nhạt
                self._uav_positions[i] * pix_square_size,
                self.uav_sensor_range * pix_square_size,
                width=1
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == '__main__':
    # Đoạn code này để kiểm tra nhanh môi trường
    print("--- Chạy kiểm tra môi trường UAV Swarm ---")

    # Cài đặt pygame nếu chưa có
    try:
        import pygame
    except ImportError:
        print("Đang cài đặt pygame để render...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])

    # Chạy với chế độ render
    env = UAVSwarmEnv(render_mode="human")
    obs, info = env.reset()

    for _ in range(200):
        # Chọn hành động ngẫu nhiên cho mỗi UAV
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)

        if reward > 0:
            print(f"Nhận được thưởng: {reward}")

        if terminated or truncated:
            print("Episode kết thúc. Đang reset...")
            obs, info = env.reset()

    env.close()
    print("--- Kiểm tra hoàn tất ---")

