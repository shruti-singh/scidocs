#!/bin/sh

#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=23:59:00
#SBATCH --job-name=specsc
#SBATCH --error=./pa_logs/specsc_0_%j.err
#SBATCH --output=./pa_logs/specsc_0_%j.log
#SBATCH --partition=small
cd $SLURM_SUBMIT_DIR 

pwd; hostname; date
source ~/.bashrc
conda activate scidocs
python 1E_embed_scratchtrained_specter.py

date
