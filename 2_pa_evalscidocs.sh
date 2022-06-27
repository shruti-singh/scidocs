#!/bin/sh

#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=6:59:00
#SBATCH --job-name=evalscidocs
#SBATCH --error=./pa_logs/evalscidocs_%j.err
#SBATCH --output=./pa_logs/evalscidocs_%j.log
#SBATCH --partition=small
cd $SLURM_SUBMIT_DIR 

pwd; hostname; date
source ~/.bashrc
conda activate scidocs
python 2_eval-scidocs.py

date
