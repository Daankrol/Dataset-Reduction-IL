# Dataset Reduction Methods for Data and Energy Efficient Deep Learning

This repository contains the code for the MSc Thesis "Dataset Reduction Methods for Data and Energy Efficient Deep Learning" by Daan Krol. It contains the code for all non-adaptive and adaptive data subset selection methods. The code is based on the [CORDS library](https://github.com/decile-team/cords). It is extended with several other dataset reduction methods, both adaptive and non-adaptive. In addition to that, it contains energy consumption tracking, feature space visualization, subset sampling visualization, biodiversity datasets, W&B logging, possibility to run with pre-trained networks and a framework for running experiments.

## Setup
To install the requirements, run the following command in the root directory of the project:

```bash
pip install -r requirements/requirements-nov.txt
```

This project was designed to be run on a high performance cluster which uses [SLURM](https://slurm.schedmd.com/overview.html). To run the code on a different cluster, the SLURM commands in the code should be replaced with the corresponding commands for the cluster. See the _slurm_job_submitter.py_ file for more information.

This project uses the Weights and Biases platform to track experiments. To use this, create a free account on the [Weights and Biases website](https://wandb.ai/). Then, run the following command to login to the Weights and Biases platform:

```bash
wandb login
```

You now only have to change the Weights and Biases configuration in the constructor of the _TrainClassifier_ class which can be found in the _train_sl.py_ file. 


## Example run

Experiments can be performed by using config files. See the _exp_configs_ directory for examples per dataset. A config can be overridden by using the command line arguments. The optimal number of workers for each dataloader differs per machine and should be optimized.

An example using the Papilionidae dataset with the GradMatchPB method while using warmup:
```bash
python3 run_experiment.py --num_workers 8 --config ./exp_configs/papilion/config_gradmatchpb.py --fraction 0.2 --select_every 10 --kappa 0.5
```

Or we can use pre-trained networks:
```bash
python3 run_experiment.py --num_workers 8 --config ./exp_configs/papilion/config_gradmatchpb.py --fraction 0.2 --select_every 10 --pretrained --finetune --lr 0.005 --kappa 0.3 --epochs 150
```

For more info on the available arguments, see the run_experiment.py file and checkout the config files.