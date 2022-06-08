# Dataset reduction using CORDS
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data

parser = argparse.ArgumentParser(description="Run experiments with config file\n ./configs/SL/conig_gradmatch_cifar10.py")
parser.add_argument(
    "--config",
    type=str,
    help="Path of the experiment config file",
)

args = parser.parse_args()
cfg = load_config_data(args.config)
clf = TrainClassifier(cfg)
clf.train() # train and evaluate