import argparse
import os
from time import sleep

# echo "GPU ID: ${SLURM_JOB_GPUS}"
# cat /proc/self/status | grep Cpus_allowed_list

# OUTPUT:
# GPU ID: 7
# Cpus_allowed_list:	39-41,49,95-97,105

def run_jobs_slurm(jobs_path: str, partition: str = None, cluster: str = 'intel'):
    """
    SLURM job runner
    Run in folder where you want to create jobs, e.g.
    `python slurm_jobs_runner.py --jobs_path jobs.txt --conda_env ote-det`
    Make sure your ~/.bashrc contains the following lines:
    export PATH="/home2/<user>/miniconda3/bin:$PATH"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home2/<user>/miniconda3/lib
    `jobs_path` should contain a list of jobs, for example
    python scripts/run_experiment2.py --task classification --dataset pap --epochs 30 --rounds -1 --runs 5 --debug --active_learning_sample_size 50 --disable_output_embedding --active_learning random
    python scripts/run_experiment2.py --task classification --dataset pap --epochs 30 --rounds -1 --runs 5 --debug --active_learning_sample_size 50 --disable_output_embedding --active_learning uncertainty_exploration
    after a job is submitted to SLURM it is commented out
    # python scripts/run_experiment2.py --task classification --dataset pap --epochs 30 --rounds -1 --runs 5 --debug --active_learning_sample_size 50 --disable_output_embedding --active_learning random # ok
    python scripts/run_experiment2.py --task classification --dataset pap --epochs 30 --rounds -1 --runs 5 --debug --active_learning_sample_size 50 --disable_output_embedding --active_learning uncertainty_exploration
    job files are created in the current directory as job<line_number>.sh, where line_number refers to `jobs_path`
    :param jobs_path: path to txt file containing jobs
    :param conda_environment: path to conda env
    :param partition: partition to submit to. E.g. 'gpu', 'gpushort'
    :param cluster: cluster to submit to, can be 'intel' or 'rug'
    """
    if partition is None:
        if cluster == 'intel':
            partition = 'short'
        else:
            partition = 'gpu'

    with open(jobs_path, "r") as f:
        lines = f.readlines()
    os.chdir(
        os.path.expanduser("~/jobs")
    )  # force .sh and .out files to be stored in ~/jobs
    for line_num, line in enumerate(lines):
        if (
                not line.startswith("#")  # comment or job already run
                and not len(line.strip()) == 0  # empty line
        ):
            line = line.strip()
            tokens = line.split("#")
            line = tokens[0].strip()
            # write SLURM bash script
            python_command = line
            job_name = f"job{line_num}"

            command = f"""#!/bin/bash
#SBATCH --job-name={job_name}"""
            if partition is not None:
                command += f"\n#SBATCH --partition={partition}"
            command += f"""
#SBATCH --nodes=1
#SBATCH --cores=12
#SBATCH --mem=100G
"""
            if cluster == 'intel':
                command += "#SBATCH --gres=gpu:1\n"
                if partition == 'short':
                    command += "#SBATCH --time=23:00:00\n"
                else:
                    command += "#SBATCH --time=1-23:00:00\n"
                command += f"""
source ~/.bashrc
source activate /home/daankrol/miniconda3/envs/DatasetReduction/
cd ~/Dataset-Reduction-IL
{python_command}"""

            elif cluster == 'rug':
                command += f"""#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=d.j.krol.1@student.rug.nl
#SBATCH --time=2:00:00

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/DatasetReduction/bin/activate
cd /data/$USER/Dataset-Reduction-IL/
{python_command}
"""
            job_sh_path = f"job{line_num}.sh"
            with open(job_sh_path, "w") as f:
                f.write(command)
            # execute the job
            success = os.system(f"sbatch {job_sh_path}")  # success: 0 is ok, 1 is error
            line = "# " + line + " # " + ("ok" if success == 0 else "error")
            lines[line_num] = line + "\n"
            with open(jobs_path, "w") as f:
                f.writelines(lines)
            sleep(2)  # prevent duplicate output folders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit jobs to SLURM")
    parser.add_argument(
        "--jobs_path",
        type=str,
        required=True,
        help="Path of a .txt file that contains a list of jobs to be submitted.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        choices=["short", "main", "gpu", "gpushort"],
        default=None,
        help="The SLURM partition to put this job in. options: \n(intel): [short, main] \n(rug): [gpu, gpushort]",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        choices=["intel", "rug"],
        default="intel",
        help="The SLURM cluster to put this job in (options: [intel, rug]).",
    )

    args = parser.parse_args()

    # if jobs_path is empty, show help mesage of argparse and exit
    if args.jobs_path is None:
        parser.print_help()
        exit(1)

    run_jobs_slurm(args.jobs_path, cluster=args.cluster, partition=args.partition)