'''
(i promise this isn't chat gpt its just how i think b4 writing 
file iterative funcs)
make a script in python

that goes through every .R  in ./bnRep/data-raw ,

and then delete the line with usethis:

and then append write.bif(accidents, file = "filename.bif") and then execute it

and move all the .bifs to ./converts
'''

from pathlib import Path
import subprocess
import os
import re

bn_dir = './bnRep/data-raw'
bn_dir = Path(bn_dir)
output_dir = './converts'

bn_dir_cd = Path('../bnRep/data-raw')


def edit_file(file):
    with file.open("r") as f:
        lines = f.readlines()
    lines = [l for l in lines if 'usethis' not in l]

    object_names = []
    for line in lines:
        match = re.match(r"^\s*(\w+)\s*<-\s*custom\.fit\(", line)
        if match:
            object_names.append(match.group(1))

    lines.insert(0, '\nlibrary(bnlearn)\n')

    for obj in object_names:
        lines.append(f'\nwrite.bif({obj}, file = "{obj}.bif")\n')

    # warn if nothing was matched
    if not object_names:
        print(f"No custom.fit() assignments found in {file.name}")

    # Write the modified content back to the file
    with file.open('w') as f:
        f.writelines(lines)

def main():

    errors = []

    for f in bn_dir.iterdir():
        edit_file(f)


    os.chdir(output_dir)

    for f in bn_dir_cd.iterdir():
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
    
    print(f'Had issues converting: {errors}')

if __name__ == "__main__":
    main()  