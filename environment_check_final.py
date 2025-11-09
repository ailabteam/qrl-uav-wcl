import sys
import torch
import pennylane as qml
import gymnasium as gym

# --- In thông tin cơ bản ---
print("="*60)
print("       BẮT ĐẦU KIỂM TRA MÔI TRƯỜNG DỰ ÁN QRL-UAV (BẢN SỬA LỖI)")
print("="*60)
print(f"Phiên bản Python: {sys.version}")
print(f"Thực thi từ: {sys.executable}")
print("-" * 60)

# --- 1. Kiểm tra PyTorch và GPU ---
print("\n[1] KIỂM TRA PYTORCH & GPU CUDA...")
try:
    print(f"   - Phiên bản PyTorch: {torch.__version__}")
    assert torch.cuda.is_available(), "PyTorch không phát hiện được CUDA. Cài đặt có thể đã sai."
    
    gpu_count = torch.cuda.device_count()
    current_gpu_idx = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu_idx)
    
    print(f"   - [THÀNH CÔNG] PyTorch phát hiện được {gpu_count} GPU.")
    print(f"   - GPU đang dùng: GPU {current_gpu_idx} - {gpu_name}")
    print(f"   - Phiên bản CUDA được PyTorch biên dịch cho: {torch.version.cuda}")

    tensor_cpu = torch.tensor([1.0, 2.0])
    tensor_gpu = tensor_cpu.to("cuda")
    result = (tensor_gpu + tensor_gpu) * 2
    print(f"   - [THÀNH CÔNG] Phép tính trên GPU thành công: {tensor_cpu.numpy()} -> {result.cpu().numpy()}")

except Exception as e:
    print(f"   - [LỖI] Xảy ra lỗi trong quá trình kiểm tra PyTorch: {e}")
    sys.exit(1)

# --- 2. Kiểm tra Pennylane và Plugin GPU ---
print("\n[2] KIỂM TRA PENNYLANE & PLUGIN GPU...")
try:
    print(f"   - Phiên bản Pennylane: {qml.__version__}")
    
    dev = qml.device("lightning.gpu", wires=1)
    print(f"   - [THÀNH CÔNG] Đã khởi tạo thiết bị 'lightning.gpu' thành công.")

    # ***** THAY ĐỔI DUY NHẤT Ở ĐÂY *****
    # Thay đổi 'backprop' thành 'adjoint' để tương thích với lightning.gpu
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def simple_quantum_circuit(inputs):
        qml.RX(inputs[0], wires=0)
        qml.RY(inputs[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    inputs_gpu = torch.tensor([0.54, 0.12], requires_grad=True, device="cuda")
    result_gpu = simple_quantum_circuit(inputs_gpu)
    print(f"   - [THÀNH CÔNG] Mạch lượng tử đã thực thi trên thiết bị: {result_gpu.device}")
    
    result_gpu.backward()
    print(f"   - [THÀNH CÔNG] Lan truyền ngược qua mạch lượng tử thành công (sử dụng phương pháp Adjoint).")
    print(f"   - Gradient tính được: {inputs_gpu.grad.cpu().numpy()}")

except Exception as e:
    print(f"   - [LỖI] Xảy ra lỗi trong quá trình kiểm tra Pennylane: {e}")
    sys.exit(1)

# --- 3. Kiểm tra Gymnasium ---
print("\n[3] KIỂM TRA GYMNASIUM...")
try:
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    print(f"   - [THÀNH CÔNG] Đã tạo môi trường Gymnasium 'CartPole-v1'.")
    print(f"   - Kích thước Quan sát: {env.observation_space.shape}")
    print(f"   - Số lượng Hành động: {env.action_space.n}")
    env.close()
except Exception as e:
    print(f"   - [LỖI] Không thể tạo môi trường Gymnasium: {e}")
    sys.exit(1)

# --- Kết luận ---
print("-" * 60)
print("\n[KẾT LUẬN] Môi trường đã được thiết lập chính xác và sẵn sàng cho dự án!")
print("Tất cả các thành phần cốt lõi (PyTorch-GPU, Pennylane-GPU, Gymnasium) đều hoạt động.")
print("="*60)
