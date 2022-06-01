# Dataset reduction using CORDS


from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data

config_file = "/cords/cords/configs/SL/config_glister_cifar10.py"
cfg = load_config_data(config_file)
print(cfg)
# cfg[""]
# clf = TrainClassifier(cfg)
# clf.train()
