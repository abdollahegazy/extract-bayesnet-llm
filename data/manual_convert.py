import os

from utils import convert_all_bnrep_networks


if __name__ == "__main__":    

    network_names  = [f.replace(".rda","") for f in os.listdir('./bnRep/data') if f.endswith('.rda')]

    network_names = ['agropastoral', 'liquefaction', 'algal2', 'aspergillus', 'consequenceCovid', 'darktriad', 'covidtech', 'stocks', 'covidfear', 'income', 'covidrisk', 'algorithms', 'turbine', 'diagnosis', 'firealarm', 'covid', 'suffocation', 'foodsecurity', 'humanitarian', 'cachexia', 'intensification', 'lexical', 'safespeed', 'projectmanagement', 'estuary', 'bank', 'student', 'lawschool', 'ricci', 'expenditure']

    convert_all_bnrep_networks(network_names)

    # convert_all_bnrep_networks(['crypto'])

