import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_detailed_comparison_plot(py_csv_path, rs_csv_path, save_path=None, figsize=(12, 7)):
    """
    Create a more detailed comparison plot with individual data points
    """
    
    # Read the CSV file
    py_df = pd.read_csv(py_csv_path)
    rs_df = pd.read_csv(rs_csv_path)
    df = pd.concat([py_df, rs_df])
    print(df.info())
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Bar chart with error bars
    pytorch_mean = df['pytorch_time_ms'].mean()
    pytorch_std = df['pytorch_time_ms'].std()
    onnx_mean = df['onnx_time_ms'].mean()
    onnx_std = df['onnx_time_ms'].std()
    rs_onnx_mean = df['rs_onnx_time_ms'].mean()
    rs_onnx_std = df['rs_onnx_time_ms'].std()
    
    models = ['PyTorch', 'Python-ONNX', "Rust-ONNX"]
    means = [pytorch_mean, onnx_mean, rs_onnx_mean]
    stds = [pytorch_std, onnx_std, rs_onnx_std]
    
    colors = ['#3498db', '#e74c3c', "#40dd10"]
    bars = ax1.bar(models, means, yerr=stds, capsize=5, 
                   color=colors, alpha=1,
                   edgecolor='black', linewidth=1.2)
    
    ax1.set_ylabel('Inference Time (ms)', fontweight='bold')
    ax1.set_title('Mean Inference Time')
    ax1.grid(True, axis='y', alpha=0.5)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height() + std
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{mean:.2f}±{std:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Box plot
    data_for_box = [
        py_df['pytorch_time_ms'].values, 
        py_df['onnx_time_ms'].values, 
        rs_df['rs_onnx_time_ms'].values
    ]
    box = ax2.boxplot(data_for_box, tick_labels=models, patch_artist=True,
                      boxprops=dict(alpha=1),
                      medianprops=dict(color='black', linewidth=2))
    
    # Color the boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Inference Time (ms)', fontweight='bold')
    ax2.set_title('Distribution Comparison')
    ax2.grid(True, axis='y', alpha=0.5)

    # Overall title
    fig.suptitle('PyTorch vs Python-ONNX vs Rust-ONNX - Isolated Inference Time [RegNet_y_16gf]', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed plot saved to: {save_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)

# Example usage
if __name__ == "__main__":

    name_base = "plot"
    base_file = "RegNet_y_16gf_benchmark_no_pre"
    py_csv_file_path = f"../data/benchmarks/python/{base_file}.csv"
    py_csv_file_path_abs = Path(py_csv_file_path).resolve()
    rs_csv_file_path = f"../data/benchmarks/rs/{base_file}.csv"
    rs_csv_file_path_abs = Path(rs_csv_file_path).resolve()
    
    # Create detailed comparison plot
    print("\nCreating detailed comparison plot...")
    create_detailed_comparison_plot(
        py_csv_file_path_abs, 
        rs_csv_file_path_abs, 
        save_path=f"../images/{base_file}_{name_base}.png"
    )
    