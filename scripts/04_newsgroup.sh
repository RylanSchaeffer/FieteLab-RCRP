#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 4                    # number of  cores
#SBATCH --mem=32G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u exp_04_newsgroup/main.py        # -u flushes output buffer immediately
