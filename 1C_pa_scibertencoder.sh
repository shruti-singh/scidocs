#!/bin/sh

#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=23:59:00
#SBATCH --job-name=scibertenc
#SBATCH --error=./pa_logs/scibertenc_%j.err
#SBATCH --output=./pa_logs/scibertenc_%j.log
#SBATCH --partition=small
cd $SLURM_SUBMIT_DIR 

pwd; hostname; date
source ~/.bashrc
conda activate scidocs
python 1C_embed_scibert.py

date
