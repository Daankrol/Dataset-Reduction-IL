#!/bin/bash
#SBATCH --time=00:59:00
#SBATCH --job-name=peregrine_test
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=d.j.krol.1@student.rug.nl
#SBATCH --array=1
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1


module purge
# module load PyTorch/1.10.0-fosscuda-2020b
module load Python/3.8.6-GCCcore-10.2.0
# On peregrine use the old dependency resolver of pip:
# python -m pip install --no-cache -r requirements/requirements-nov.txt

source /data/$USER/.envs/DatasetReduction/bin/activate
# do a wandb relogin with api key 
# export WANDB_API_KEY=a3c0932a7937da444ca4c32cc2a14a4d20c0649f
cd /data/$USER/Dataset-Reduction-IL/
python main.py