import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_detailed_comparison_plot(save_path=None, figsize=(12, 7)):
    """
    Create a more detailed comparison plot with individual data points
    """
    
    # Create figure with subplots
    fig, (ax1) = plt.subplots(1, 1, figsize=figsize)
    
    # Values in ms
    pytorch_mean = 3214
    pytorch_std =  178
    onnx_mean = 262.3
    onnx_std = 9.2
    rs_onnx_mean = 91.4
    rs_onnx_std = 2.4
    
    models = ['PyTorch', 'Python-ONNX', "Rust-ONNX"]
    means = [pytorch_mean, onnx_mean, rs_onnx_mean]
    stds = [pytorch_std, onnx_std, rs_onnx_std]
    
    colors = ['#3498db', '#e74c3c', "#40dd10"]
    bars = ax1.bar(models, means, yerr=stds, capsize=5, 
                   color=colors, alpha=1,
                   edgecolor='black', linewidth=1.2)
    
    ax1.set_ylabel('Inference Time (ms)', fontweight='bold')
    ax1.set_title('Mean Execution Time')
    ax1.grid(True, axis='y', alpha=0.5)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height() + std
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{mean:.2f}±{std:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Overall title
    fig.suptitle('PyTorch vs Python-ONNX vs Rust-ONNX: Hyperfine Comparison [RegNet_x_400mf]', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed plot saved to: {save_path}")
    
    plt.show()
    
    return fig, (ax1)

# Example usage
if __name__ == "__main__":

    name_base = "RegNet_x_400mf"

    # Create detailed comparison plot
    print("\nCreating detailed comparison plot...")
    create_detailed_comparison_plot(
        save_path=f"../images/{name_base}_hyperfine_comparison.png"
    )
    