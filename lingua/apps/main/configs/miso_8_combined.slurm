#!/bin/bash

#SBATCH --account=miso
#SBATCH --job-name=LM8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1438G
#SBATCH --partition=miso,miso-lo
#SBATCH --output=out/log.out
#SBATCH --error=out/log.err
#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block
#SBATCH --mail-type=BEGIN,END,FAIL

# Mimic the effect of "conda init", which doesn't work for scripts
eval "$($CONDA_PATH shell.bash hook)"
source activate $CONDA_ENV_PATH

mkdir -p out/logs

export OMP_NUM_THREADS=1
export LAUNCH_WITH="SBATCH"
