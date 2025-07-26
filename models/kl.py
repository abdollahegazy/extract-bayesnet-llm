import numpy as np
from itertools import product
from pgmpy.inference import VariableElimination


def kl_bn_local(bn_p, bn_q):
    """
    Calculate KL divergence between two Bayesian networks using local decomposition.
    
    D_KL(p || q) = Σᵢ Σ_paᵢ p(paᵢ) D_KL(p(Xᵢ|paᵢ) || q(Xᵢ|paᵢ))
    """
    # Get CPDs as dictionaries
    cpds_p = {cpd.variable: cpd for cpd in bn_p.get_cpds()}
    cpds_q = {cpd.variable: cpd for cpd in bn_q.get_cpds()}
    
    # Check structure match
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
    eps = 1e-12  # Small constant for numerical stability
    
    for var_name in cpds_p.keys():
        cpd_p = cpds_p[var_name]
        cpd_q = cpds_q[var_name]
        parents = cpd_p.get_evidence()
        
        if not parents:
            # No parents: p(paᵢ) = 1
            p_vals = cpd_p.values.copy()
            q_vals = cpd_q.values.copy()
            
            # Add epsilon to avoid log(0)
            q_vals = np.maximum(q_vals, eps)
            
            # Ensure normalization
            p_vals = p_vals / p_vals.sum()
            q_vals = q_vals / q_vals.sum()
            
            # Calculate KL divergence
            mask = p_vals > 0
            kl_inner = np.sum(p_vals[mask] * np.log(p_vals[mask] / q_vals[mask]))
            kl_total += kl_inner
            
        else:
            # Has parents: need to compute p(paᵢ)
            marginals = infer_p.query(variables=list(parents), show_progress=False)
            
            # CRITICAL: The order of parents in cpd.variables[1:] is the order used in cpd.values
            # This may differ from cpd.get_evidence()
            cpd_all_vars = cpd_p.variables  # [variable, parent1, parent2, ...]
            cpd_parent_order_in_values = cpd_all_vars[1:]  # Parents in the order they appear in values array
            
            # Create a mapping from parent name to its index in the CPD values array
            parent_to_cpd_axis = {parent: i + 1 for i, parent in enumerate(cpd_parent_order_in_values)}
            
            # Get the order of variables in marginals
            marginal_vars = marginals.scope()
            
            # Get cardinalities in marginal's order
            marginal_cards = [marginals.get_cardinality([var])[var] for var in marginal_vars]
            
            for marginal_idx in product(*[range(c) for c in marginal_cards]):
                # Get p(paᵢ)
                ppa_i = marginals.values[marginal_idx]
                
                # Build the slicer based on CPD's actual parent order
                slicer = [slice(None)]  # First dimension is for the variable itself
                
                # Add indices for each parent in the order they appear in CPD values
                for parent in cpd_parent_order_in_values:
                    # Find this parent's index in the marginal
                    marginal_pos = marginal_vars.index(parent)
                    parent_state = marginal_idx[marginal_pos]
                    slicer.append(parent_state)
                
                slicer = tuple(slicer)
                
                try:
                    p_vals = cpd_p.values[slicer].copy()
                    q_vals = cpd_q.values[slicer].copy()
                except IndexError as e:
                    print(f"Error for {var_name}:")
                    print(f"  CPD shape: {cpd_p.values.shape}")
                    print(f"  CPD variables: {cpd_p.variables}")
                    print(f"  CPD cardinality: {cpd_p.cardinality}")
                    print(f"  Parents from get_evidence(): {list(parents)}")
                    print(f"  Parents from cpd.variables: {cpd_parent_order_in_values}")
                    print(f"  Marginal variables: {marginal_vars}")
                    print(f"  Marginal index: {marginal_idx}")
                    print(f"  Trying to slice with: {slicer}")
                    raise e
                
                # Add epsilon to avoid log(0)
                q_vals = np.maximum(q_vals, eps)
                
                # Ensure normalization
                p_vals = p_vals / p_vals.sum()
                q_vals = q_vals / q_vals.sum()
                
                # Calculate KL divergence for this parent configuration
                mask = p_vals > 0
                kl_inner = np.sum(p_vals[mask] * np.log(p_vals[mask] / q_vals[mask]))
                
                # Weight by p(paᵢ)
                kl_total += ppa_i * kl_inner
    
    return kl_total