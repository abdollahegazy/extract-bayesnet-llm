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
        ("./MLE/1000.txt", "MLE-1000"),
        ("./MLE/300.txt", "MLE-300"), 
        ("./MLE/100.txt", "MLE-100"),
        
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

    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#3b923a']
    
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
    
    # print("\nData Statistics:")
    # for i, (data, label) in enumerate(zip(all_data, labels)):
    #     median_val = np.median(data)
    #     mean_val = np.mean(data)
    #     std_val = np.std(data)
    #     print(f"{label}: N={len(data)}, Median={median_val:.3f}, Mean={mean_val:.3f}, Std={std_val:.3f}")
    
    plt.savefig('bn_kl_divergence_boxplot.png', dpi=300, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":

    create_bn_kl_boxplot()
