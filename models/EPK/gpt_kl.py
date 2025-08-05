import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #for import KL


if __name__ == "__main__":
    from kl import kl_bn_local
    from pprint import pprint
    from pgmpy.readwrite import BIFReader
    from pathlib import Path
    import sys
    model_name = sys.argv[1]

    kls = []
    e=[]

    og_bns = sorted(Path('../../data/filtered_networks').iterdir())

    for og_bn in og_bns:
        name = og_bn.stem
        gpt_file = Path(f'./4omini/{name}.bif')
        if not gpt_file.exists():continue

        og_bn = BIFReader(og_bn).get_model()
        gpt_bn = BIFReader(gpt_file).get_model()

        try:

            div = kl_bn_local(og_bn,gpt_bn)

            kls.append(div)
            print(f"KL DIV: {div:.2f}\n")
                # break

        except Exception as error: e.append(name);print(error);continue;
    
    if e:
        pprint(e)
        pprint(f'had {len(e)} error networks')
    else:
        print('ran with no errors.')

    with open(f'{model_name}.txt','w') as f:
        for kl in kls:
            f.write(f'{str(kl)}\n') 


    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create boxplot with custom styling
    box_plot = ax.boxplot(kls, 
                         patch_artist=True,  # Enable fill colors
                         medianprops={'color': 'black', 'linewidth': 2},
                         whiskerprops={'color': 'black', 'linewidth': 1.5},
                         capprops={'color': 'black', 'linewidth': 1.5},
                         flierprops={'marker': 'o', 'markerfacecolor': 'red', 
                                   'markersize': 5, 'markeredgecolor': 'black'})
    
    # Color the box dark blue
    for patch in box_plot['boxes']:
        patch.set_facecolor('darkblue')
        patch.set_edgecolor('black')
    
    # Calculate and display quantiles
    q1, median, q3,q4 = np.percentile(kls, [25, 50, 75,100])
    min_val, max_val = np.min(kls), np.max(kls)
    
    # Add quantile labels
    # ax.text(1.15, q1+2, f'Q1: {q1:.2f}', va='center', fontsize=10, 
            # bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    ax.text(1.15, median+2, f'Median: {median:.2f}', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    # ax.text(1.15, q4, f'Q4: {q4:.2f}', va='center', fontsize=10,
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    # Set y-axis properties
    ax.set_ylabel('BN KL Divergence Distribution', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 14)
    ax.set_yticks(np.arange(0, 14.5, 0.5))
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Set x-axis properties
    ax.set_xlabel(f'4o-mini EPK', fontsize=12, fontweight='bold')
    ax.set_xticklabels([f'4o-mini EPK'])
    
    # Add title
    ax.set_title(f'KL Divergence Distribution for 4omini)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    # plt.savefig(f"../images/4ominiepk.png", dpi=300, bbox_inches='tight')
    plt.show()

