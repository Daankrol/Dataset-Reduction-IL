# Dataset reduction using CORDS
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data
import argparse
import os
import os.path as osp
from pyJoules.energy_meter import measure_energy, EnergyMeter, EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.device_factory import DeviceFactory



def getCPUIDs():
    """Get the IDs of the GPU and CPUs that are assigned to this job. 
    SLURM IDs are mapped to the psysical IDs.
    """
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

    return cpu_id_list

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

parser.add_argument(
    "--gpu",
    type=int,
    help="GPU id"
)

args = parser.parse_args()

cfg = load_config_data(args.config)
clf = TrainClassifier(cfg)

if cfg.measure_energy:
    cpu_id_list = getCPUGPUIDs()
    print(f"Running with GPU {args.gpu} and CPU IDs: {cpu_id_list} ")

    domains = [RaplPackageDomain(id) for id in cpu_id_list] + [NvidiaGPUDomain(args.gpu)]

    results_dir = osp.abspath(osp.expanduser(cfg.train_args.results_dir))
    if cfg.dss_args.type != "Full":
        all_logs_dir = os.path.join(results_dir, cfg.setting,
                                    cfg.dss_args.type,
                                    cfg.dataset.name,
                                    str(cfg.dss_args.fraction),
                                    str(cfg.dss_args.select_every))
    else:
        all_logs_dir = os.path.join(results_dir, cfg.setting,
                                    cfg.dss_args.type,
                                    cfg.dataset.name)

    os.makedirs(all_logs_dir, exist_ok=True)
    csv_handler = CSVHandler(all_logs_dir + "/energy_consumption.csv")

    with EnergyContext(handler=csv_handler, domains=domains) as meter:
        clf.train() # train and evaluate

    csv_handler.save_data()

else:
    clf.train()