#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 4                    # number of  cores
#SBATCH --mem=64G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u exp_03_reddit/main.py        # -u flushes output buffer immediately
