#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # two cores
#SBATCH --mem=10G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u exp_01_mixture_of_gaussians/main.py        # -u flushes output buffer immediately
