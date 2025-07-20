import os

from utils import convert_all_bnrep_networks


if __name__ == "__main__":    

    network_names  = [f.replace(".rda","") for f in os.listdir('./bnRep/data') if f.endswith('.rda')]
    convert_all_bnrep_networks(network_names)

    # convert_all_bnrep_networks(['crypto'])

