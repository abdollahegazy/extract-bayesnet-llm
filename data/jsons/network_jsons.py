import json
from pathlib import Path
from pgmpy.readwrite import BIFReader

def get_bn_structure_json(bn,name):
    """
    Extract nodes and their possible states from a Bayesian Network
    and return as JSON format
    """
    nodes_info = {}
    
    # Get all CPDs from the network
    cpds = bn.get_cpds()
    
    for cpd in cpds:
        node_name = cpd.variable
    
        states = cpd.state_names[node_name]
        
        nodes_info[node_name] = {
            "states": states,
            # "number of states": int(cpd.cardinality[0]),
        }

    json.dump(nodes_info,open(f'./data/{name}.json',"w"),indent=2)
    return nodes_info

# Example usage:
if __name__ == "__main__":
    # Load a Bayesian Network
    BN_PATH = Path('../filtered_networks')
    for bn in BN_PATH.iterdir():
        bn_name = bn.stem
        bn = BIFReader(bn).get_model()
        get_bn_structure_json(bn,bn_name)
