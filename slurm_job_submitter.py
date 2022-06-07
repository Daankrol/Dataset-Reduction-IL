import argparse
import os
from time import sleep


def run_jobs_slurm(jobs_path: str, conda_environment: str, partition: str = None, cluster: str = 'Intel'):
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
    :param conda_environment:
    """
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
            command += f"""\n#SBATCH --nodes=1
#SBATCH --cores=7
#SBATCH --gres=gpu:1"""
            if partition == 'Intel':
                command += f"\n#SBATCH --mem=100G"
            else:
                command += f"\n#SBATCH --mem=20G"
            command += """\n#SBATCH --mem=100G
source ~/.bashrc
source activate {conda_environment}
cd ~/ResearchFramework
export TMPDIR=/home/$USER/tmp
export PYTHONPATH=$PYTHONPATH:.:third_party/training_extensions/external/mmdetection
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
        help="Path of a .txt file that contains a list of jobs to be submitted.",
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        default=None,
        help="The existing conda virtual environment to activate for this job.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        choices=["short", "main"],
        default=None,
        help="The SLURM partition to put this job in (options: [short, main])",
    )

    args = parser.parse_args()

    run_jobs_slurm(args.jobs_path, args.conda_env, partition=args.partition)