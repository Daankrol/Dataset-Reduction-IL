# Dataset reduction using CORDS
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data
import argparse


def getCPUGPUIDs():
    """Get the IDs of the GPU and CPUs that are assigned to this job. 
    SLURM IDs are mapped to the psysical IDs.
    """
    # GPU ID can be found using the environment variable SLURM_JOB_GPU
    gpu_id = os.environ.get("SLURM_JOB_GPU")

    # CPU IDs can be found by using the bash command "cat /proc/self/status | grep Cpus_allowed_list"
    # This will return a string with the format "Cpus_allowed_list:	39-41,49,95-97,105"
    # For IDS that are separated by a '-' we also want the ids that are in between the '-'

    cpu_id_string = os.popen("cat /proc/self/status | grep Cpus_allowed_list").read()
    cpu_id_string = cpu_id_string.split(":")[1].strip()
    cpu_id_list = []
    for cpu_id_string in cpu_id_string.split(","):
        if "-" in cpu_id_string:
            cpu_id_list.extend(range(int(cpu_id_string.split("-")[0]), int(cpu_id_string.split("-")[1]) + 1))
        else:
            cpu_id_list.append(int(cpu_id_string))

    # The mapping from logical to physical IDs can be found using the bash command 'lstopo --only pu'
    # This will return a multi-line string with the format: 'PU L#1 (P#56)'
    # The PU L# is the logical ID and the P# is the physical ID

    mapping_str = os.popen("lstopo --only pu").read()
    mapping_str = mapping_str.split("\n")
    mapping = {}
    for line in mapping_str:
        if "PU L#" in line:
            mapping[line.split(" ")[1].split("#")[1]] = line.split(" ")[2].split("#")[1].split(')')[0]

    for cpu_id in cpu_id_list:
        if cpu_id in mapping:
            cpu_id_list[cpu_id_list.index(cpu_id)] = mapping[cpu_id]

    return gpu_id, cpu_id_list

parser = argparse.ArgumentParser(description="Run experiments with config file\n ./configs/SL/config_gradmatch_cifar10.py")
parser.add_argument(
    "--config",
    type=str,
    help="Path of the experiment config file",
)

parser.add_argument(
    "--cluster",
    type=str,
    choices=["intel", "rug"],
    help="Either 'intel' or 'rug'"
)



args = parser.parse_args()

if args.cluster == "intel":
    gpu_id, cpu_id_list = getCPUGPUIDs()

print(f"Running with GPU {gpu_id} and CPU IDs: {cpu_id_list} ")
# cfg = load_config_data(args.config)
# clf = TrainClassifier(cfg)
# clf.train() # train and evaluate