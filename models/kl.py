import numpy as np
from itertools import product
from pgmpy.inference import VariableElimination


def kl_bn_local(bn_p, bn_q):
    """
    Calculate KL divergence between two Bayesian networks using local decomposition.
    
    Parameters:
    bn1, bn2: BayesianNetwork objects (same structure)
    
    Returns:
    float: KL divergence D_KL(bn1 || bn2)
    """
    kl_total = 0
    
    # Get CPDs as dictionaries for easy lookup
    cpds_p = {cpd.variable: cpd for cpd in bn_p.get_cpds()}
    cpds_q = {cpd.variable: cpd for cpd in bn_q.get_cpds()}
    

    vars_p = set(cpds_p)
    vars_q = set(cpds_q)
    if vars_p != vars_q:
        missing_in_q = vars_p - vars_q
        missing_in_p = vars_q - vars_p
        raise ValueError(
            f"Structure mismatch: "
            f"P has {missing_in_q} not in Q; "
            f"Q has {missing_in_p} not in P"
        )
    

    infer_p = VariableElimination(bn_p)

    kl_total = 0
    eps = 0

    for var_name in cpds_p.keys():
        # print(f"{var_name}: {cpds_p[var_name].values}")
        cpd_p = cpds_p[var_name]
        cpd_q = cpds_q[var_name]
        parents = cpd_p.get_evidence()  # Now cpd_p is the actual CPD object   


        if not parents:

            ppa_i = 1.0
            p_vals = cpd_p.values
            q_vals = cpd_q.values

            mask = p_vals > 0   

            q_vals[q_vals == 0] = 1e-12      # only adjust zeros    

            q_vals /= q_vals.sum()
            p_vals /= p_vals.sum()


            kl_inner = np.sum(p_vals[mask] * np.log(p_vals[mask] / q_vals[mask]))

            kl_total+= ppa_i*kl_inner
        
        else:
            marginals = infer_p.query(variables=parents,show_progress=False)
            
            scope = marginals.scope()
            cardinality_dict = marginals.get_cardinality(scope)  
            cards = [cardinality_dict[p] for p in parents]

            for parent_idx in product(*[range(c) for c in cards]):
                ppa_i = marginals.values[parent_idx]

                slicer = (slice(None),) + parent_idx

                p_vals = cpd_p.values[slicer]  
                q_vals = cpd_q.values[slicer] 

                p_vals /= p_vals.sum()   

                q_vals[q_vals == 0] = 1e-12      # only adjust zeros    
                q_vals /= q_vals.sum()

                mask = p_vals > 0   



                kl_inner = np.sum(p_vals[mask] * np.log(p_vals[mask] / q_vals[mask]))
        
                kl_total+=ppa_i*kl_inner

    return kl_total

        

