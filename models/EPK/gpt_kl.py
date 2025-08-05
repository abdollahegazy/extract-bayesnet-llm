import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #for import KL


if __name__ == "__main__":
    from kl import kl_bn_local
    from pprint import pprint
    from pgmpy.readwrite import BIFReader
    from pathlib import Path
    import sys
    model_name = sys.argv[1]

    kls = []
    e=[]

    og_bns = sorted(Path('../../data/filtered_networks').iterdir())

    for og_bn in og_bns:
        name = og_bn.stem
        gpt_file = Path(f'./4omini/{name}.bif')
        if not gpt_file.exists():continue

        og_bn = BIFReader(og_bn).get_model()
        gpt_bn = BIFReader(gpt_file).get_model()

        try:

            div = kl_bn_local(og_bn,gpt_bn)

            kls.append(div)
            print(f"KL DIV: {div:.2f}\n")
                # break

        except Exception as error: e.append(name);print(error);continue;
    
    if e:
        pprint(e)
        pprint(f'had {len(e)} error networks')
    else:
        print('ran with no errors.')

    with open(f'{model_name}.txt','w') as f:
        for kl in kls:
            f.write(f'{str(kl)}\n') 