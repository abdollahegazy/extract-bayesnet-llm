
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
        