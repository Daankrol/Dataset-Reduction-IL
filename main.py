# Dataset reduction using CORDS
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data

print("test!")
config_file = "./configs/SL/config_glister_cifar10.py"
cfg = load_config_data(config_file)
print(cfg)
clf = TrainClassifier(cfg)
clf.train()