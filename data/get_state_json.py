def get_bn_structure_json(bn,name):
    import json

    nodes_info = {}
    cpds = bn.get_cpds()

    for cpd in cpds:

        node_name = cpd.variable
        states = cpd.state_names[node_name]

        nodes_info[node_name] = {
            "states": states,
        }

    json.dump(nodes_info,open(f'./jsons/{name}.json',"w"),indent=2)
    return nodes_info

if __name__ == "__main__":
    from pgmpy.readwrite import BIFReader
    from pathlib import Path


    BN_PATH = Path('./filtered_networks')
    for bn in BN_PATH.iterdir():
        bn_name = bn.stem
        bn = BIFReader(bn).get_model()
        get_bn_structure_json(bn,bn_name)
