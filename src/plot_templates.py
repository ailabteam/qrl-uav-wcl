# file: src/plot_templates.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .publication_style import CONTEXT_COLORS

def _setup_ax_and_save(ax, figsize):
    """Helper để quản lý việc tạo axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        return fig, ax, True # save_and_close = True
    else:
        fig = ax.get_figure()
        return fig, ax, False # save_and_close = False

def plot_line_comparison(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    smoothing_window: int = 50,
    figsize: tuple = (6, 4),
    ax=None
):
    """
    Vẽ biểu đồ đường so sánh từ dữ liệu dạng 'long'.
    """
    fig, ax, save_and_close = _setup_ax_and_save(ax, figsize)

    agents = data[hue_col].unique()
    
    colors = {
        'QRL (Proposed)': CONTEXT_COLORS['proposed'],
        'DRL (Baseline)': CONTEXT_COLORS['baseline']
    }
    linestyles = {
        'QRL (Proposed)': '-',
        'DRL (Baseline)': '--'
    }

    for agent in agents:
        agent_data = data[data[hue_col] == agent]
        
        # Áp dụng bộ lọc làm mịn (rolling average)
        smoothed_y = agent_data[y_col].rolling(window=smoothing_window, min_periods=1).mean()

        ax.plot(
            agent_data[x_col],
            smoothed_y,
            label=agent,
            color=colors.get(agent, 'black'),
            linestyle=linestyles.get(agent, ':')
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title='Agent Type')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    if save_and_close:
        plt.savefig(output_path)
        print(f"Line plot saved to: {output_path}")
        plt.close(fig)
    return ax

def plot_grouped_bar_chart(
    data: pd.DataFrame,
    category_col: str,
    value_cols: list,
    y_label: str,
    title: str,
    output_path: str,
    figsize: tuple = (7, 5),
    ax=None
):
    """
    Vẽ biểu đồ cột nhóm.
    """
    fig, ax, save_and_close = _setup_ax_and_save(ax, figsize)

    # Chuyển đổi dữ liệu sang dạng phù hợp để vẽ
    df_plot = data.set_index(category_col)[value_cols]
    
    n_categories = len(df_plot.index)
    n_values = len(df_plot.columns)
    
    x = np.arange(n_categories)
    width = 0.8 / n_values
    
    colors = [CONTEXT_COLORS.get(c) for c in ['proposed', 'baseline']]

    for i, value_col in enumerate(df_plot.columns):
        offset = width * (i - (n_values - 1) / 2)
        measurements = df_plot[value_col]
        rects = ax.bar(x + offset, measurements, width, label=value_col, color=colors[i % len(colors)])
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x, df_plot.index)
    ax.legend(title='Agent')
    ax.set_ylim(0, df_plot.max().max() * 1.15) # Tự động điều chỉnh trục y
    ax.grid(axis='x', which='both', visible=False)

    if save_and_close:
        plt.savefig(output_path)
        print(f"Grouped bar chart saved to: {output_path}")
        plt.close(fig)
    return ax
