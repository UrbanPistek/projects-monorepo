import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_timing_comparison_plot(csv_path, save_path=None, figsize=(9, 6)):
    """
    Create a bar chart comparing PyTorch and ONNX inference times with error bars
    
    Args:
        csv_path: Path to the CSV file
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Calculate statistics
    pytorch_mean = df['pytorch_time_ms'].mean()
    pytorch_std = df['pytorch_time_ms'].std()
    onnx_mean = df['onnx_time_ms'].mean()
    onnx_std = df['onnx_time_ms'].std()
    
    # Create the data for plotting
    models = ['PyTorch', 'ONNX']
    means = [pytorch_mean, onnx_mean]
    stds = [pytorch_std, onnx_std]
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the bar plot
    bars = ax.bar(models, means, yerr=stds, capsize=8, 
                  color=['#3498db', '#e74c3c'], alpha=0.8, 
                  edgecolor='black', linewidth=1.2)
    
    # Customize the plot
    ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Model Inference Time Comparison\nPyTorch vs ONNX Runtime', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on top of bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height() + std
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{mean:.2f}±{std:.2f}ms', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Calculate speedup
    speedup = pytorch_mean / onnx_mean
    
    # Add speedup annotation
    ax.text(0.98, 0.98, f'ONNX Speedup: {speedup:.2f}×', 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            ha='right', va='top')
    
    # Add sample size info
    ax.text(0.02, 0.98, f'Sample size: {len(df)} images', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            ha='left', va='top')
    
    # Customize grid and spines
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Add some padding to the top
    y_max = max(means[i] + stds[i] for i in range(len(means)))
    ax.set_ylim(top=y_max * 1.2)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TIMING COMPARISON SUMMARY")
    print("="*50)
    print(f"PyTorch - Mean: {pytorch_mean:.3f}ms, Std: {pytorch_std:.3f}ms")
    print(f"ONNX    - Mean: {onnx_mean:.3f}ms, Std: {onnx_std:.3f}ms")
    print(f"ONNX Speedup: {speedup:.2f}×")
    print(f"Time Reduction: {((pytorch_mean - onnx_mean) / pytorch_mean * 100):.1f}%")
    print(f"Sample Size: {len(df)} images")
    
    return fig, ax


def create_detailed_comparison_plot(csv_path, save_path=None, figsize=(9, 6)):
    """
    Create a more detailed comparison plot with individual data points
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Bar chart with error bars
    pytorch_mean = df['pytorch_time_ms'].mean()
    pytorch_std = df['pytorch_time_ms'].std()
    onnx_mean = df['onnx_time_ms'].mean()
    onnx_std = df['onnx_time_ms'].std()
    
    models = ['PyTorch', 'ONNX']
    means = [pytorch_mean, onnx_mean]
    stds = [pytorch_std, onnx_std]
    
    bars = ax1.bar(models, means, yerr=stds, capsize=5, 
                   color=['#3498db', '#e74c3c'], alpha=1,
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
    data_for_box = [df['pytorch_time_ms'].values, df['onnx_time_ms'].values]
    box = ax2.boxplot(data_for_box, tick_labels=models, patch_artist=True,
                      boxprops=dict(alpha=1),
                      medianprops=dict(color='black', linewidth=2))
    
    # Color the boxes
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Inference Time (ms)', fontweight='bold')
    ax2.set_title('Distribution Comparison')
    ax2.grid(True, axis='y', alpha=0.5)

    # Overall title
    fig.suptitle('PyTorch vs ONNX Runtime - Python Inference Time', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed plot saved to: {save_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)

# Example usage
if __name__ == "__main__":

    name_base = "py_torch_v_onnx_inference_regnet_MX_400F"
    csv_file_path = "../data/benchmarks/benchmark_batch.csv"
    csv_file_path_abs = Path(csv_file_path).resolve()
    
    # Create simple comparison plot
    print("Creating simple comparison plot...")
    create_timing_comparison_plot(csv_file_path_abs, save_path=f"../images/{name_base}_timing_comparison.png")
    
    # Create detailed comparison plot
    print("\nCreating detailed comparison plot...")
    create_detailed_comparison_plot(csv_file_path_abs, save_path=f"../images/{name_base}_detailed_comparison.png")
    