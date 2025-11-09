# file: src/publication_style.py
import matplotlib.pyplot as plt

COLOR_PALETTE = {
    'blue':   '#1f77b4',
    'orange': '#ff7f0e',
    'green':  '#2ca02c',
    'red':    '#d62728',
    'purple': '#9467bd',
    'gray':   '#7f7f7f',
}

CONTEXT_COLORS = COLOR_PALETTE.copy()
CONTEXT_COLORS.update({
    'proposed':   CONTEXT_COLORS['red'],
    'baseline':   CONTEXT_COLORS['gray'],
    'method_A':   CONTEXT_COLORS['blue'],
    'method_B':   CONTEXT_COLORS['green'],
})

def set_publication_style(font_family='sans-serif'):
    """
    Thiết lập các thông số rcParams của Matplotlib cho figure chất lượng cao.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Font Settings
    if font_family == 'sans-serif':
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    else: # serif
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    # Font Size
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # Line and Marker Settings
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 5
    
    # Axes and Ticks Settings
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    
    # Legend Settings
    plt.rcParams['legend.frameon'] = False
    
    # Figure Saving Settings
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.format'] = 'pdf' # PDF là định dạng vector, tốt nhất cho publication
    plt.rcParams['savefig.bbox'] = 'tight'
    
    print("Publication style set successfully.")
