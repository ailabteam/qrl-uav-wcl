import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

# Thiết lập thiết bị lượng tử. Chúng ta sẽ sử dụng lightning.gpu để có hiệu suất cao nhất.
# wires sẽ được xác định khi tạo lớp mô hình.
dev = qml.device("lightning.gpu", wires=4) # Giả định tạm thời có 4 qubit

def create_quantum_circuit(n_qubits, n_layers):
    """
    Hàm helper để tạo một nút lượng tử (QNode).
    QNode này định nghĩa kiến trúc của Variational Quantum Circuit (VQC).
    """
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def quantum_circuit(inputs, weights):
        """
        :param inputs: Dữ liệu đầu vào, được mã hóa vào các qubit. Kích thước (n_qubits,).
        :param weights: Các tham số có thể học của mạch. Kích thước (n_layers, n_qubits, 3).
        """
        # 1. Mã hóa dữ liệu đầu vào (Embedding)
        # Sử dụng AngleEmbedding để mã hóa các features thành các góc quay của qubit.
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        # 2. Các lớp biến đổi (Variational Layers)
        # Sử dụng các lớp lặp lại để tăng khả năng biểu diễn của mạch.
        for i in range(n_layers):
            # Lớp các cổng quay có tham số
            for j in range(n_qubits):
                qml.RX(weights[i, j, 0], wires=j)
                qml.RY(weights[i, j, 1], wires=j)
                qml.RZ(weights[i, j, 2], wires=j)
            
            # Lớp các cổng làm rối (Entangling Layer)
            # Sử dụng CNOT để tạo ra sự vướng víu lượng tử giữa các qubit.
            for j in range(n_qubits - 1):
                qml.CNOT(wires=[j, j + 1])
            if n_qubits > 1:
                 qml.CNOT(wires=[n_qubits - 1, 0]) # Tạo vướng víu vòng

        # 3. Phép đo (Measurement)
        # Trả về giá trị kỳ vọng của toán tử PauliZ trên mỗi qubit.
        # Điều này cho chúng ta một vector đầu ra có kích thước n_qubits.
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return quantum_circuit

class QuantumCritic(nn.Module):
    """
    Một mạng Critic lai Lượng tử-Cổ điển.
    Kiến trúc: [Lớp cổ điển đầu vào] -> [Lớp lượng tử] -> [Lớp cổ điển đầu ra]
    """
    def __init__(self, state_dim, action_dim, n_qubits=4, n_layers=2):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Các lớp cổ điển
        # Lớp tiền xử lý để giảm chiều dữ liệu xuống n_qubits
        self.pre_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits)
        )

        # Lớp lượng tử
        q_circuit = create_quantum_circuit(n_qubits, n_layers)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(q_circuit, weight_shapes)

        # Lớp hậu xử lý để tạo ra Q-value cuối cùng
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output là một giá trị Q duy nhất
        )

    def forward(self, state, action):
        # Nối trạng thái và hành động lại với nhau
        x = torch.cat([state, action], dim=1)
        
        # Đưa qua lớp tiền xử lý
        x = self.pre_net(x)

        # Đưa qua lớp lượng tử
        x = self.q_layer(x)

        # Đưa qua lớp hậu xử lý để nhận Q-value
        q_value = self.post_net(x)
        
        return q_value

if __name__ == '__main__':
    # Đoạn code kiểm tra nhanh
    print("--- Chạy kiểm tra mô hình Quantum Critic ---")
    state_dim = 10
    action_dim = 2
    
    # Giả lập dữ liệu đầu vào trên GPU
    batch_size = 4
    dummy_state = torch.randn(batch_size, state_dim).to("cuda")
    dummy_action = torch.randn(batch_size, action_dim).to("cuda")
    
    # Tạo mô hình và chuyển nó lên GPU
    q_critic = QuantumCritic(state_dim, action_dim).to("cuda")
    
    # Thực hiện một lượt lan truyền xuôi
    q_values = q_critic(dummy_state, dummy_action)
    
    print(f"Kích thước đầu vào state: {dummy_state.shape}")
    print(f"Kích thước đầu vào action: {dummy_action.shape}")
    print(f"Kích thước đầu ra Q-value: {q_values.shape}")
    assert q_values.shape == (batch_size, 1)
    
    # Kiểm tra lan truyền ngược
    target = torch.randn(batch_size, 1).to("cuda")
    loss = nn.MSELoss()(q_values, target)
    loss.backward()
    
    print(f"Giá trị loss: {loss.item()}")
    print("Lan truyền ngược thành công!")
    
    # In ra số lượng tham số
    total_params = sum(p.numel() for p in q_critic.parameters() if p.requires_grad)
    print(f"Tổng số tham số có thể huấn luyện: {total_params}")

    print("--- Kiểm tra hoàn tất ---")
