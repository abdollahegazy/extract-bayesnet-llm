from pathlib import Path

OUTPUT_DIR = './raw/automatic_converts'
BN_REP_RAW_DATA = Path('../bnRep/data-raw')


def edit_file(file):
    import re
    
    with file.open("r") as f:
        lines = f.readlines()
    #some weird R thing i don't care about
    lines = [l for l in lines if 'usethis' not in l]

    object_names = []
    for line in lines:
        #regex to get BN name from R script on line like: BOPfailure2 <- custom.fit(kick.dag,cpt)
        match = re.match(r"^\s*(\w+)\s*<-\s*custom\.fit\(", line)
        if match:
            object_names.append(match.group(1))

    #make sure bnlearn is loaded in R script
    lines.insert(0, '\nlibrary(bnlearn)\n')

    #append line to write BN to BIF (this only works for discrete BNs)
    for obj in object_names:
        lines.append(f'\nwrite.bif({obj}, file = "{obj}.bif")\n')


    if not object_names:
        print(f"No custom.fit() assignments found in {file.name}")

    with file.open('w') as f:
        f.writelines(lines)

def main():
    import os
    import subprocess


    errors = []
    
    #edit Rscripts to include lines for saving BN
    for f in BN_REP_RAW_DATA.iterdir():
        edit_file(f)

    os.chdir(OUTPUT_DIR)

    for f in BN_REP_RAW_DATA.iterdir():
        #run modified Rscript to save BN
        try:
            subprocess.run(
                ['Rscript',f],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)
            print(f"{f.name} ran successfully.")
        except subprocess.CalledProcessError as e:
            print(f'Could not convert {f.name} b/c {e.stderr}')
            errors.append(f.stem)
    
    print(f'[ERROR] Had issues converting: {errors}')

if __name__ == "__main__":
    main()  