#!/bin/sh

#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=23:59:00
#SBATCH --job-name=sbertenc
#SBATCH --error=./pa_logs/sbertenc_%j.err
#SBATCH --output=./pa_logs/sbertenc_%j.log
#SBATCH --partition=small
cd $SLURM_SUBMIT_DIR 

pwd; hostname; date
source ~/.bashrc
conda activate scidocs
python 1D_embed_sbert.py

date
