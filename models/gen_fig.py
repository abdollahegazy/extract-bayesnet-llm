import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_data_from_file(filepath):
    with open(filepath, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return data


def create_bn_kl_boxplot():
    data_sources = [
        ("./MLE/TEMP_1000.txt", "MLE-1000"),
        ("./MLE/TEMP_300.txt", "MLE-300"), 
        ("./MLE/TEMP_100.txt", "MLE-100"),
        
        # EPK data
        ("./EPK/4omini.txt", "GPT-4o-mini (EPK)"),
    ]
    

    all_data = []
    labels = []
    
    for filepath, label in data_sources:
        data = load_data_from_file(filepath)
        all_data.append(data)
        labels.append(label)
        print(f"Loaded {len(data)} values from {filepath}")
    

    plt.figure(figsize=(16, 8))

    box_plot = plt.boxplot(all_data, labels=labels, patch_artist=True)
    
    for median in box_plot['medians']:
        median.set_color('black')
        median.set_linewidth(0.75)    

    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
              '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5']
    
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Distribution of BN KL Divergence of Bayesian Networks by Model',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('BN KL Divergence Distribution', fontsize=12)
    plt.xlabel('')
    
    plt.xticks(rotation=45, ha='right')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.ylim(0, 14)
    
    plt.tight_layout()
    
    print("\nData Statistics:")
    for i, (data, label) in enumerate(zip(all_data, labels)):
        median_val = np.median(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        print(f"{label}: N={len(data)}, Median={median_val:.3f}, Mean={mean_val:.3f}, Std={std_val:.3f}")
    
    plt.savefig('bn_kl_divergence_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create the boxplot
    create_bn_kl_boxplot()
    
    # Optional: Print directory structure to help debug file paths
    print("\nCurrent directory structure:")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.txt'):
                print(f"{subindent}{file}")