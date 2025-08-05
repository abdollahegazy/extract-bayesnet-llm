
import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING) #annoying warnings to ignore

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #for import KL


def mle_baseline(model,N):
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator

    #ensure all states are covered
    state_names = {
        var: model.get_cpds(var).state_names[var]
        for var in model.nodes()
    }

    #do forward samplin
    samples = model.simulate(n_samples=N,seed=42) #,seed=240)


    mle_model = DiscreteBayesianNetwork()
    mle_model.add_nodes_from(model.nodes())
    mle_model.add_edges_from(model.edges())

    #do MLE
    mle_model.fit(
        data=samples,
        estimator=MaximumLikelihoodEstimator,
        state_names=state_names,
    )
    return mle_model



if __name__ == "__main__":
    from pgmpy.readwrite import BIFReader
    from kl import kl_bn_local
    from pathlib import Path
    import sys
    
    kls = []
    e=[]

    n_samples = int(sys.argv[1])

    
    for bn_name in Path('../../data/filtered_networks').iterdir():
        
        try:
            print("="*20,bn_name,"="*20)
            bn = BIFReader(bn_name).get_model()
            mle_model = mle_baseline(bn, N=n_samples)
            div = kl_bn_local(bn,mle_model)

            kls.append(div)
            print(f"KL DIV: {div:.2f}\n")
        except: print(f'{bn_name.stem} had issue');e.append(bn_name);pass

    if e:
        print(f'[ERROR] Had some issues with {e}')
    else:
        print('Finished with no errors')

    with open(f'TEMP_{n_samples}.txt','w') as f:
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
    ax.set_xlabel(f'MLE-N{n_samples}', fontsize=12, fontweight='bold')
    ax.set_xticklabels([f'MLE-N{n_samples}'])
    
    # Add title
    ax.set_title(f'KL Divergence Distribution for MLE Baseline (N={n_samples})', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    # plt.savefig(f"../images/TEMP_MLE_{n_samples}.png", dpi=300, bbox_inches='tight')
    plt.show()

