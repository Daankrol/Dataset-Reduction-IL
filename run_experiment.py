# Dataset reduction using CORDS
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data
import argparse
import os
from dotmap import DotMap
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
    print(cpu_id_string)
    cpu_id_string = cpu_id_string.split(":")[1].strip()
    cpu_id_list = []
    for cpu_id_string in cpu_id_string.split(","):
        if "-" in cpu_id_string:
            cpu_id_list.extend(
                range(
                    int(cpu_id_string.split("-")[0]),
                    int(cpu_id_string.split("-")[1]) + 1,
                )
            )
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
            mapping[line.split(" ")[1].split("#")[1]] = (
                line.split(" ")[2].split("#")[1].split(")")[0]
            )

    for cpu_id in cpu_id_list:
        if cpu_id in mapping:
            cpu_id_list[cpu_id_list.index(cpu_id)] = mapping[cpu_id]

    return cpu_id_list


parser = argparse.ArgumentParser(
    description="Run experiments with config file\n ./configs/SL/config_gradmatch_cifar10.py"
)

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path of the experiment config file",
)

parser.add_argument(
    "--name",
    type=str,
    help="Name of the W&B run",
)

parser.add_argument(
    "--fraction", type=float, help="Fraction of data to select with DSS"
)
parser.add_argument(
    "--kappa", type=float, help="Fraction of epochs to use for warm up."
)
parser.add_argument(
    "--select_every", type=int, help="Select a new subset every X epochs."
)
parser.add_argument("--epochs", type=int, help="number of epochs to train")
parser.add_argument(
    "--disable_scheduler",
    action='store_true',
    help="Whether to disable Cosine Annealing",
)
parser.add_argument(
    "--lam", type=float, help="regularization constant for OMP solver (GradMatch)"
)
parser.add_argument(
    "--early_stopping",
    action='store_true',
    help="Enable early stopping. NOTE: has to be used with --disable_scheduler",
)
parser.add_argument(
    "--lr",
    type=float,
    help="learning rate"
)
parser.add_argument(
    "--model",
    type=str,
    choices=["ResNet18", "EfficientNet"],
)
parser.add_argument(
    "--pretrained",
    action='store_true',
    help="use pretrained model"
)
parser.add_argument(
    "--finetune",
    action='store_true',
    help="do finetuning of all layers."
)
parser.add_argument(
    "--selection_type",
    type=str,
    help="Selection type for DSS strategy"
)
parser.add_argument(
    "--nonadaptive",
    action='store_true',
    help='Whether to run the DSS nonadaptively. (only once at the start)'
)

parser.add_argument(
    "--submod_function",
    type=str,
    choices=["facility-location", "graph-cut", "sum-redundancy", "saturated-coverage"]
)

parser.add_argument(
    '--data_dependent_scheduler',
    action='store_true',
    help='Enable data dependent scheduling of LR'
)

parser.add_argument(
    '--num_workers',
    type=int,
    help="Number of workers for dataloader"
)
parser.add_argument(
    '--batch_size',
    type=int,
    help="Batch size for dataloader"
)

args = parser.parse_args()
if args.config is None:
    parser.print_help()
    exit(1)

cfg = load_config_data(args.config)

if args.name is not None:
    cfg.name = args.name

if args.lr is not None:
    cfg.optimizer.lr = args.lr
if args.model is not None:
    cfg.model.architecture = args.model
if args.selection_type is not None:
    cfg.dss_args.selection_type = args.selection_type
if args.submod_function is not None:
    cfg.dss_args.submod_func_type = args.submod_function
if args.pretrained:
    cfg.model.type = 'pre-trained'

if args.finetune:
    cfg.model.fine_tune = args.finetune
if args.data_dependent_scheduler:
    cfg.scheduler.data_dependent = args.data_dependent_scheduler

if args.fraction is not None:
    cfg.dss_args.fraction = args.fraction
if args.select_every is not None:
    cfg.dss_args.select_every = args.select_every
if args.epochs is not None:
    cfg.train_args.num_epochs = args.epochs
    cfg.scheduler.T_max = args.epochs
if args.disable_scheduler:
    cfg.scheduler.type = None
if args.early_stopping:
    cfg.early_stopping = True
    cfg.scheduler.type = None
if args.kappa is not None:
    cfg.dss_args.kappa = args.kappa
if args.lam is not None:
    cfg.dss_args.lam = args.lam

if args.nonadaptive:
    cfg.dss_args.online = False
elif cfg.dss_args.online is None or cfg.dss_args.online == DotMap():
    cfg.dss_args.online = True

if args.num_workers is not None:
    cfg.num_workers = args.num_workers
if args.batch_size is not None:
    cfg.dataloader.batch_size = args.batch_size

clf = TrainClassifier(cfg)
clf.train()
