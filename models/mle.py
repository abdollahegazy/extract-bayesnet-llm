
import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)
from pathlib import Path
from pprint import pprint

from kl import kl_bn_local
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.readwrite import BIFReader


def mle_baseline(model: DiscreteBayesianNetwork, N: int) -> DiscreteBayesianNetwork:

    state_names = {
        var: model.get_cpds(var).state_names[var]
        for var in model.nodes()
    }

    samples = model.simulate(n_samples=N,seed=240)

    mle_model = DiscreteBayesianNetwork()
    mle_model.add_nodes_from(model.nodes())
    mle_model.add_edges_from(model.edges())


    mle_model.fit(
        data=samples,
        estimator=MaximumLikelihoodEstimator,
        state_names=state_names,
    )
    return mle_model



if __name__ == "__main__":
    
    kls = []

    e=[]
    samples = 100

    for bn in Path('../data/temp').iterdir():
        try:
            print("="*20,bn,"="*20)

            bn = BIFReader(bn).get_model()
            mle_model = mle_baseline(bn, N=samples)
            div = kl_bn_local(bn,mle_model)

            kls.append(div)
            print(f"Acheived {div} KL Divergence\n")

        except: pass;e.append(bn)
            
    pprint(e)
    pprint(f'had {len(e)} error networks')

    import matplotlib.pyplot as plt
    plt.boxplot(kls)
    plt.ylim(top=14)
    plt.savefig(f"./images/MLE_{samples}")
