# file: run_analysis.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Thêm 'src' vào path để import được các module của chúng ta
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from publication_style import set_publication_style
from plot_templates import plot_line_comparison #, plot_grouped_bar_chart (sẽ dùng sau)

def main():
    # --- Bước 1: Thiết lập Style ---
    set_publication_style()

    # --- Bước 2: Vẽ Figure 1 - Learning Curve ---
    log_dir = os.path.join(os.path.dirname(__file__), 'results', 'logs')
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    log_files_info = {
        'QRL (Proposed)': os.path.join(log_dir, 'QRL_log.csv'),
        'DRL (Baseline)': os.path.join(log_dir, 'DRL_log.csv')
    }
    
    all_data_frames = []
    for agent_name, filepath in log_files_info.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Agent'] = agent_name
            all_data_frames.append(df)
        else:
            print(f"Cảnh báo: Không tìm thấy file log '{filepath}'.")
    
    if not all_data_frames:
        print("Không có dữ liệu log để vẽ Figure 1.")
        return

    combined_df = pd.concat(all_data_frames, ignore_index=True)
    
    plot_line_comparison(
        data=combined_df,
        x_col='Total Timesteps',
        y_col='Episode Reward',
        hue_col='Agent',
        title='UAV Swarm Task Performance Comparison',
        xlabel='Training Timesteps',
        ylabel='Smoothed Episode Reward',
        output_path=os.path.join(output_dir, 'fig1_learning_curve.pdf')
    )
    
    # --- Bước 3: Vẽ Figure 2 - Final Performance (sẽ làm sau) ---
    # print("\nBỏ qua Figure 2 cho đến khi có dữ liệu đánh giá.")
    # TODO: Thêm logic để load model đã huấn luyện, chạy evaluation,
    # và vẽ biểu đồ cột so sánh các metric.

if __name__ == "__main__":
    main()
