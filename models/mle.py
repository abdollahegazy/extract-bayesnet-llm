import numpy as np
import pandas as pd
from kl import kl_bn
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.readwrite import BIFReader
from pathlib import Path

def mle_baseline(model: DiscreteBayesianNetwork, N: int) -> DiscreteBayesianNetwork:
    """
    Create MLE baseline by sampling from ground truth and re-learning parameters.
    
    Parameters:
    gt_model: Ground truth Bayesian network
    N: Number of samples to generate
    
    Returns:
    DiscreteBayesianNetwork: MLE-fitted network with same structure
    """

    state_names = {
        var: bn.get_cpds(var).state_names[var]
        for var in bn.nodes()
    }
    samples = model.simulate(n_samples=N)

    new_model = DiscreteBayesianNetwork(ebunch=model.edges())

    new_model.fit(
        data=samples,
        estimator=MaximumLikelihoodEstimator,
        state_names=state_names
    )
    return new_model




# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create MLE baseline
    kls = []

    for bn_name in Path('../data/filtered_networks').iterdir():
        bn = BIFReader(bn_name).get_model()
        mle_model = mle_baseline(bn, N=100)

        s = kl_bn(bn,mle_model)
        kls.append(s)

    plt.hist(kls)
    plt.savefig("temp")
