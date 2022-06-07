#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --job-name=peregrine_test
#SBATCH --mem=40G
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=d.j.krol.1@student.rug.nl
#SBATCH --array=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40:1


module purge
module load PyTorch/1.10.0-fosscuda-2020b
# On peregrine use the old dependency resolver of pip:
# python -m pip install --use-feature=2020-resolver -r requirements/requirements.txt
# module load Python/3.8.6-GCCcore-10.2.0

source /data/$USER/.envs/DatasetReduction/bin/activate

# do a wandb relogin with api key 
cd /data/$USER/Dataset-Reduction-IL/
python main.py