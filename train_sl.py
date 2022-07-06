import logging
import os
import os.path as osp
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dotmap import DotMap
from ray import tune
from torch.utils.data import Subset
from cords.utils.config_utils import load_config_data
from cords.utils.data.data_utils import WeightedSubset
from cords.utils.data.data_utils import collate
from cords.utils.data.dataloader.SL.adaptive import (
    GLISTERDataLoader,
    OLRandomDataLoader,
    CRAIGDataLoader,
    GradMatchDataLoader,
    RandomDataLoader,
    SELCONDataLoader,
    UncertaintyDataLoader,
)

from cords.utils.data.dataloader.SL.nonadaptive import FacLocDataLoader
from cords.utils.data.datasets.SL import gen_dataset
from cords.utils.models import *
from cords.utils.data.data_utils.collate import *
import pickle
import wandb
from pyJoules.energy_meter import EnergyMeter
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.print_handler import PrintHandler
from pyJoules.device.device_factory import DeviceFactory
import torchmetrics

from cords.utils.utils import EarlyStopping


class TrainClassifier:
    def __init__(self, config_file_data):
        self.cfg = config_file_data
        results_dir = osp.abspath(osp.expanduser(self.cfg.train_args.results_dir))

        if self.cfg.dss_args.type != "Full":
            all_logs_dir = os.path.join(
                results_dir,
                self.cfg.setting,
                self.cfg.dss_args.type,
                self.cfg.dataset.name,
                str(self.cfg.dss_args.fraction),
                str(self.cfg.dss_args.select_every),
            )
        else:
            all_logs_dir = os.path.join(
                results_dir,
                self.cfg.setting,
                self.cfg.dss_args.type,
                self.cfg.dataset.name,
            )

        os.makedirs(all_logs_dir, exist_ok=True)
        # setup logger
        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )
        log_level = logging.INFO
        if self.cfg.logging == "DEBUG":
            log_level = logging.DEBUG
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(plain_formatter)
        s_handler.setLevel(log_level)
        self.logger.addHandler(s_handler)
        f_handler = logging.FileHandler(
            os.path.join(
                all_logs_dir,
                self.cfg.dataset.name + "_" + self.cfg.dss_args.type + ".log",
            )
        )
        f_handler.setFormatter(plain_formatter)
        f_handler.setLevel(log_level)
        self.logger.addHandler(f_handler)
        self.logger.propagate = False

        if self.cfg.wandb:
            name = self.cfg.dss_args.type + "_" + self.cfg.dataset.name
            if self.cfg.dss_args.fraction != DotMap():
                name += f"_{str(self.cfg.dss_args.fraction)}"
            if self.cfg.dss_args.select_every != DotMap():
                name += f"_{str(self.cfg.dss_args.select_every)}"
            if self.cfg.model.type == "pre-trained":
                name += "_PT"
            if self.cfg.model.fine_tune != DotMap() and self.cfg.model.fine_tune:
                name += "_FT"
            if self.cfg.early_stopping:
                name += "_ES"
            if self.cfg.scheduler.type is None and not self.cfg.early_stopping:
                name += "_NoSched"
            if self.cfg.dss_args.kappa != DotMap() and self.cfg.dss_args.kappa > 0:
                name += f"_k-{str(self.cfg.dss_args.kappa)}"
            if self.cfg.dss_args.lam != DotMap() and self.cfg.dss_args.lam > 0:
                name += f"_lam-{str(self.cfg.dss_args.lam)}"
            wandb.init(
                project="Dataset Reduction for IL",
                entity="daankrol",
                name=name,
                config={
                    "dataset": self.cfg.dataset.name,
                    "dss_type": self.cfg.dss_args.type,
                    "fraction": self.cfg.dss_args.fraction,
                    "selection_type": self.cfg.dss_args.selection_type,
                    "class_imbalance_training": self.cfg.dss_args.valid,
                    "select_every": self.cfg.dss_args.select_every,
                    "setting": self.cfg.setting,
                    "model": self.cfg.model.architecture,
                    "model_type": self.cfg.model.type,
                    "fine_tune": self.cfg.model.fine_tune,
                    "epochs": self.cfg.train_args.num_epochs,
                    "batch_size": self.cfg.dataloader.batch_size,
                    "lr": self.cfg.optimizer.lr,
                    "optimizer_type": self.cfg.optimizer.type,
                    "scheduler": self.cfg.scheduler.type,
                    "measure_energy": self.cfg.measure_energy,
                    "optimizer": self.cfg.optimizer,
                    "dss_args": self.cfg.dss_args,
                },
            )

        if self.cfg.measure_energy:
            domains = [NvidiaGPUDomain(0)]
            devices = DeviceFactory.create_devices(domains)
            self.energy_meter = EnergyMeter(devices)
            self.energy_log_handler = PrintHandler()
            self.total_energy = 0.0

    """
    ############################## Loss Evaluation ##############################
    """

    def model_eval_loss(self, data_loader, model, criterion):
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.cfg.train_args.device), targets.to(
                    self.cfg.train_args.device, non_blocking=True
                )
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    """
    ############################## Model Creation ##############################
    """

    def create_model(self):
        print(f"using model: {self.cfg.model.architecture} with {self.cfg.model}")
        if self.cfg.model.architecture == "EfficientNet":
            if self.cfg.model.type == "pre-trained":
                fine_tune = False
                if self.cfg.model.fine_tune != DotMap() and self.cfg.model.fine_tune:
                    fine_tune = True

                model = EfficientNetB0_PyTorch(
                    num_classes=self.cfg.model.numclasses,
                    pretrained=True,
                    fine_tune=fine_tune,
                )
            else:
                model = EfficientNetB0_PyTorch(
                    num_classes=self.cfg.model.numclasses, pretrained=False
                )

        elif self.cfg.model.architecture == "RegressionNet":
            model = RegressionNet(self.cfg.model.input_dim)
        elif self.cfg.model.architecture == "ResNet18":
            model = ResNet18(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == "MnistNet":
            model = MnistNet()
        elif self.cfg.model.architecture == "ResNet164":
            model = ResNet164(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == "MobileNet":
            model = MobileNet(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == "MobileNetV2":
            model = MobileNetV2(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == "MobileNet2":
            model = MobileNet2(output_size=self.cfg.model.numclasses)
        elif self.cfg.model.architecture == "HyperParamNet":
            model = HyperParamNet(self.cfg.model.l1, self.cfg.model.l2)
        elif self.cfg.model.architecture == "ThreeLayerNet":
            model = ThreeLayerNet(
                self.cfg.model.input_dim,
                self.cfg.model.numclasses,
                self.cfg.model.h1,
                self.cfg.model.h2,
            )
        elif self.cfg.model.architecture == "LSTM":
            model = LSTMClassifier(
                self.cfg.model.numclasses,
                self.cfg.model.wordvec_dim,
                self.cfg.model.weight_path,
                self.cfg.model.num_layers,
                self.cfg.model.hidden_size,
            )
        else:
            raise NotImplementedError
        model = model.to(self.cfg.train_args.device)
        return model

    """
    ############################## Loss Type, Optimizer and Learning Rate Scheduler ##############################
    """

    def loss_function(self):
        if self.cfg.loss.type == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
            criterion_nored = nn.CrossEntropyLoss(reduction="none")

        elif self.cfg.loss.type == "MeanSquaredLoss":
            criterion = nn.MSELoss()
            criterion_nored = nn.MSELoss(reduction="none")
        return criterion, criterion_nored

    def optimizer_with_scheduler(self, model):
        if self.cfg.optimizer.type == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.cfg.optimizer.lr,
                momentum=self.cfg.optimizer.momentum,
                weight_decay=self.cfg.optimizer.weight_decay,
                nesterov=self.cfg.optimizer.nesterov,
            )
        elif self.cfg.optimizer.type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.cfg.optimizer.lr)
        elif self.cfg.optimizer.type == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=self.cfg.optimizer.lr)

        if self.cfg.scheduler.type == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.scheduler.T_max
            )
        elif self.cfg.scheduler.type == "linear_decay":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.cfg.scheduler.stepsize,
                gamma=self.cfg.scheduler.gamma,
            )
        else:
            scheduler = None
        return optimizer, scheduler

    @staticmethod
    def generate_cumulative_timing(mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return mod_cum_timing

    @staticmethod
    def save_ckpt(state, ckpt_path):
        torch.save(state, ckpt_path)

    @staticmethod
    def load_ckpt(ckpt_path, model, optimizer):
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss = checkpoint["loss"]
        metrics = checkpoint["metrics"]
        return start_epoch, model, optimizer, loss, metrics

    def count_pkl(self, path):
        if not osp.exists(path):
            return -1
        return_val = 0
        file = open(path, "rb")
        while True:
            try:
                _ = pickle.load(file)
                return_val += 1
            except EOFError:
                break
        file.close()
        return return_val

    def report_energy(self, epoch=None):
        if not self.cfg.measure_energy:
            return
        self.energy_meter.stop()
        energy_samples = self.energy_meter.get_trace()
        for sample in energy_samples:
            for device in sample.energy:
                self.total_energy += sample.energy[device]
            if epoch != None:
                wandb.log(
                    {
                        "energy": sample.energy,
                        "duration": sample.duration,
                        "cumulative_energy": self.total_energy,
                    },
                    step=epoch,
                )
            else:
                wandb.log(
                    {
                        f"energy": sample.energy,
                        "duration": sample.duration,
                        "cumulative_energy": self.total_energy,
                    }
                )

    def start_energy_measurement(self, tag):
        if not self.cfg.measure_energy:
            return
        self.energy_meter.start(tag)

    def train(self):
        """
        ############################## General Training Loop with Data Selection Strategies ##############################
        """
        # Loading the Dataset
        logger = self.logger
        logger.info(self.cfg)

        # Start energy measurement
        self.start_energy_measurement(tag="data_loading")

        use_pre_trained_normalization = False
        if self.cfg.model.type == "pre-trained":
            use_pre_trained_normalization = True

        if self.cfg.dataset.feature == "classimb":
            trainset, validset, testset, num_cls = gen_dataset(
                self.cfg.dataset.datadir,
                self.cfg.dataset.name,
                self.cfg.dataset.feature,
                classimb_ratio=self.cfg.dataset.classimb_ratio,
                dataset=self.cfg.dataset,
                pre_trained=use_pre_trained_normalization,
                img_size=self.cfg.dataset.img_size,
            )
        else:
            trainset, validset, testset, num_cls = gen_dataset(
                self.cfg.dataset.datadir,
                self.cfg.dataset.name,
                self.cfg.dataset.feature,
                dataset=self.cfg.dataset,
                pre_trained=use_pre_trained_normalization,
                img_size=self.cfg.dataset.img_size,
            )

        trn_batch_size = self.cfg.dataloader.batch_size
        val_batch_size = self.cfg.dataloader.batch_size
        tst_batch_size = self.cfg.dataloader.batch_size

        if (
            self.cfg.dataset.name == "sst2_facloc"
            and self.count_pkl(self.cfg.dataset.ss_path) == 1
            and self.cfg.dss_args.type == "FacLoc"
        ):
            self.cfg.dss_args.type = "Full"
            file_ss = open(self.cfg.dataset.ss_path, "rb")
            ss_indices = pickle.load(file_ss)
            file_ss.close()
            trainset = torch.utils.data.Subset(trainset, ss_indices)

        batch_sampler = lambda _, __: None
        drop_last = False

        if "collate_fn" in self.cfg.dataloader.keys():
            collate_fn = self.cfg.dataloader.collate_fn
        else:
            collate_fn = None

        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=trn_batch_size,
            sampler=batch_sampler(trainset, trn_batch_size),
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

        valloader = torch.utils.data.DataLoader(
            validset,
            batch_size=val_batch_size,
            sampler=batch_sampler(validset, val_batch_size),
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=tst_batch_size,
            sampler=batch_sampler(testset, tst_batch_size),
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

        substrn_losses = list()  # np.zeros(cfg['train_args']['num_epochs'])
        trn_losses = list()
        val_losses = list()  # np.zeros(cfg['train_args']['num_epochs'])
        tst_losses = list()
        subtrn_losses = list()
        timing = list()
        total_timing = 0.0
        trn_acc = list()
        val_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])
        tst_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])
        subtrn_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])
        trn_recalls = list()
        val_recalls = list()
        tst_recalls = list()
        trn_precisions = list()
        val_precisions = list()
        tst_precisions = list()
        trn_f1s = list()
        val_f1s = list()
        tst_f1s = list()

        # Checkpoint file
        checkpoint_dir = osp.abspath(osp.expanduser(self.cfg.ckpt.dir))
        ckpt_dir = os.path.join(
            checkpoint_dir,
            self.cfg.setting,
            self.cfg.dss_args.type,
            self.cfg.dataset.name,
            str(self.cfg.dss_args.fraction),
            str(self.cfg.dss_args.select_every),
        )
        checkpoint_path = os.path.join(ckpt_dir, "model.pt")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Model Creation
        self.cfg.model.numclasses = num_cls
        model = self.create_model()

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        if self.cfg.wandb:
            wandb.watch(model, criterion, log="all")

        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(model)

        # Early stopping
        if self.cfg.early_stopping and scheduler is not None:
            print(self.cfg.early_stopping, scheduler)
            raise Exception("Do not use early stopping AND a lr scheduler")
        if self.cfg.early_stopping:
            early_stopping = EarlyStopping(patience=15, min_delta=0, logger=logger)

        """
        ############################## Custom Dataloader Creation ##############################
        """

        if "collate_fn" not in self.cfg.dss_args:
            self.cfg.dss_args.collate_fn = None

        if self.cfg.dss_args.type in [
            "GradMatch",
            "GradMatchPB",
            "GradMatch-Warm",
            "GradMatchPB-Warm",
        ]:
            """
            ############################## GradMatch Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device

            dataloader = GradMatchDataLoader(
                trainloader,
                valloader,
                self.cfg.dss_args,
                logger,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=self.cfg.dataloader.shuffle,
                pin_memory=self.cfg.dataloader.pin_memory,
                collate_fn=self.cfg.dss_args.collate_fn,
            )

        elif self.cfg.dss_args.type in [
            "GLISTER",
            "GLISTER-Warm",
            "GLISTERPB",
            "GLISTERPB-Warm",
        ]:
            """
            ############################## GLISTER Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device
            dataloader = GLISTERDataLoader(
                trainloader,
                valloader,
                self.cfg.dss_args,
                logger,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=self.cfg.dataloader.shuffle,
                pin_memory=self.cfg.dataloader.pin_memory,
                collate_fn=self.cfg.dss_args.collate_fn,
            )

        elif self.cfg.dss_args.type in [
            "CRAIG",
            "CRAIG-Warm",
            "CRAIGPB",
            "CRAIGPB-Warm",
        ]:
            """
            ############################## CRAIG Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device

            dataloader = CRAIGDataLoader(
                trainloader,
                valloader,
                self.cfg.dss_args,
                logger,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=self.cfg.dataloader.shuffle,
                pin_memory=self.cfg.dataloader.pin_memory,
                collate_fn=self.cfg.dss_args.collate_fn,
            )

        elif self.cfg.dss_args.type in ["Random", "Random-Warm"]:
            """
            ############################## Random Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = RandomDataLoader(
                trainloader,
                self.cfg.dss_args,
                logger,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=self.cfg.dataloader.shuffle,
                pin_memory=self.cfg.dataloader.pin_memory,
                collate_fn=self.cfg.dss_args.collate_fn,
            )

        elif self.cfg.dss_args.type == ["OLRandom", "OLRandom-Warm"]:
            """
            ############################## OLRandom Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = OLRandomDataLoader(
                trainloader,
                self.cfg.dss_args,
                logger,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=self.cfg.dataloader.shuffle,
                pin_memory=self.cfg.dataloader.pin_memory,
                collate_fn=self.cfg.dss_args.collate_fn,
            )

        elif self.cfg.dss_args.type == "FacLoc":
            """
            ############################## Facility Location Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.model = model
            self.cfg.dss_args.data_type = self.cfg.dataset.type

            dataloader = FacLocDataLoader(
                trainloader,
                valloader,
                self.cfg.dss_args,
                logger,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=self.cfg.dataloader.shuffle,
                pin_memory=self.cfg.dataloader.pin_memory,
                collate_fn=self.cfg.dss_args.collate_fn,
            )
            if (
                self.cfg.dataset.name == "sst2_facloc"
                and self.count_pkl(self.cfg.dataset.ss_path) < 1
            ):

                ss_indices = dataloader.subset_indices
                file_ss = open(self.cfg.dataset.ss_path, "wb")
                try:
                    pickle.dump(ss_indices, file_ss)
                except EOFError:
                    pass
                file_ss.close()

        elif self.cfg.dss_args.type == "Full":
            """
            ############################## Full Dataloader Additional Arguments ##############################
            """
            wt_trainset = WeightedSubset(
                trainset, list(range(len(trainset))), [1] * len(trainset)
            )

            dataloader = torch.utils.data.DataLoader(
                wt_trainset,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=self.cfg.dataloader.shuffle,
                pin_memory=self.cfg.dataloader.pin_memory,
                collate_fn=self.cfg.dss_args.collate_fn,
            )

        elif self.cfg.dss_args.type in ["SELCON"]:
            """
            ############################## SELCON Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.lr = self.cfg.optimizer.lr
            self.cfg.dss_args.loss = criterion_nored  # doubt: or criterion
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.optimizer = optimizer
            self.cfg.dss_args.criterion = criterion
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.batch_size = self.cfg.dataloader.batch_size

            # todo: not done yet
            self.cfg.dss_args.delta = torch.tensor(self.cfg.dss_args.delta)
            # self.cfg.dss_args.linear_layer = self.cfg.dss_args.linear_layer # already there, check glister init
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = SELCONDataLoader(
                trainset,
                validset,
                trainloader,
                valloader,
                self.cfg.dss_args,
                logger,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=self.cfg.dataloader.shuffle,
                pin_memory=self.cfg.dataloader.pin_memory,
            )

        elif self.cfg.dss_args.type in ["Uncertainty"]:
            """
            ############################## Uncertainty Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            dataloader = UncertaintyDataLoader(
                trainloader, valloader, self.cfg.dss_args, logger
            )

        else:
            raise NotImplementedError

        if self.cfg.dss_args.type in ["SELCON"]:
            is_selcon = True
        else:
            is_selcon = False

        self.report_energy()

        """
        ################################################# Checkpoint Loading #################################################
        """

        if self.cfg.ckpt.is_load:
            start_epoch, model, optimizer, ckpt_loss, load_metrics = self.load_ckpt(
                checkpoint_path, model, optimizer
            )
            logger.info(
                "Loading saved checkpoint model at epoch: {0:d}".format(start_epoch)
            )
            for arg in load_metrics.keys():
                if arg == "val_loss":
                    val_losses = load_metrics["val_loss"]
                if arg == "val_acc":
                    val_acc = load_metrics["val_acc"]
                if arg == "tst_loss":
                    tst_losses = load_metrics["tst_loss"]
                if arg == "tst_acc":
                    tst_acc = load_metrics["tst_acc"]
                if arg == "trn_loss":
                    trn_losses = load_metrics["trn_loss"]
                if arg == "trn_acc":
                    trn_acc = load_metrics["trn_acc"]
                if arg == "subtrn_loss":
                    subtrn_losses = load_metrics["subtrn_loss"]
                if arg == "subtrn_acc":
                    subtrn_acc = load_metrics["subtrn_acc"]
                if arg == "time":
                    timing = load_metrics["time"]
        else:
            start_epoch = 0

        """
        ################################################# Training Loop #################################################
        """

        for epoch in range(start_epoch, self.cfg.train_args.num_epochs):
            self.start_energy_measurement(tag="training")
            subtrn_loss = 0
            subtrn_correct = 0
            subtrn_total = 0
            model.train()
            start_time = time.time()
            cum_weights = 0
            for _, data in enumerate(dataloader):
                if is_selcon:
                    (
                        inputs,
                        targets,
                        _,
                        weights,
                    ) = data  # dataloader also returns id in case of selcon algorithm
                else:
                    inputs, targets, weights = data
                inputs = inputs.to(self.cfg.train_args.device)
                targets = targets.to(self.cfg.train_args.device, non_blocking=True)
                weights = weights.to(self.cfg.train_args.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                losses = criterion_nored(outputs, targets)
                if self.cfg.is_reg:
                    loss = torch.dot(losses.view(-1), weights / (weights.sum()))
                else:
                    loss = torch.dot(losses, weights / (weights.sum()))
                loss.backward()
                subtrn_loss += loss.item() * weights.sum()
                cum_weights += weights.sum()
                optimizer.step()
                if not self.cfg.is_reg:
                    if is_selcon:
                        predicted = outputs  # linaer regression in selcon
                    else:
                        _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
            epoch_time = time.time() - start_time
            if cum_weights != 0:
                subtrn_loss = subtrn_loss / cum_weights
            if not scheduler == None:
                scheduler.step()
            timing.append(epoch_time)
            total_timing += epoch_time
            print_args = self.cfg.train_args.print_args

            """
            ################################################# Evaluation Loop #################################################
            """

            if ((epoch + 1) % self.cfg.train_args.print_every == 0) or (
                epoch == self.cfg.train_args.num_epochs - 1
            ):
                trn_loss = 0
                trn_correct = 0
                trn_total = 0
                val_loss = 0
                val_correct = 0
                val_total = 0
                tst_correct = 0
                tst_total = 0
                tst_loss = 0
                model.eval()

                if ("trn_loss" in print_args) or ("trn_acc" in print_args):
                    samples = 0
                    with torch.no_grad():
                        for _, data in enumerate(trainloader):
                            if is_selcon:
                                inputs, targets, _ = data
                            else:
                                inputs, targets = data

                            inputs, targets = inputs.to(
                                self.cfg.train_args.device
                            ), targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            trn_loss += loss.item() * trainloader.batch_size
                            samples += targets.shape[0]
                            if "trn_acc" in print_args:
                                if is_selcon:
                                    predicted = outputs
                                else:
                                    _, predicted = outputs.max(1)
                                trn_total += targets.size(0)
                                trn_correct += predicted.eq(targets).sum().item()

                        trn_loss = trn_loss / samples
                        trn_losses.append(trn_loss)

                    if "trn_acc" in print_args:
                        trn_acc.append(trn_correct / trn_total)
                    if "trn_recall" in print_args:
                        # Do macro averaging recall over all classes
                        recall = torchmetrics.Recall(
                            average="macro", num_classes=self.cfg.model.numclasses
                        ).to(self.cfg.train_args.device)
                        trn_recall = recall(outputs, targets).item()
                        trn_recalls.append(trn_recall)

                    # calculate precision and f1
                    precision = torchmetrics.Precision(
                        average="macro", num_classes=self.cfg.model.numclasses
                    ).to(self.cfg.train_args.device)
                    trn_precision = precision(outputs, targets).item()
                    trn_precisions.append(trn_precision)
                    f1 = torchmetrics.F1Score(
                        average="macro", num_classes=self.cfg.model.numclasses
                    ).to(self.cfg.train_args.device)
                    trn_f1 = f1(outputs, targets).item()
                    trn_f1s.append(trn_f1)

                    # confusion matrix
                    wandb.log(
                        {
                            "trn_confusion_matrix": wandb.plot.confusion_matrix(
                                probs=outputs,
                                y_true=targets.cpu().numpy(),
                            )
                        },
                        step=epoch,
                    )

                if ("val_loss" in print_args) or ("val_acc" in print_args):
                    samples = 0
                    with torch.no_grad():
                        for _, data in enumerate(valloader):
                            if is_selcon:
                                inputs, targets, _ = data
                            else:
                                inputs, targets = data

                            inputs, targets = inputs.to(
                                self.cfg.train_args.device
                            ), targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item() * valloader.batch_size
                            samples += targets.shape[0]
                            if "val_acc" in print_args:
                                if is_selcon:
                                    predicted = outputs
                                else:
                                    _, predicted = outputs.max(1)
                                val_total += targets.size(0)
                                val_correct += predicted.eq(targets).sum().item()

                        val_loss = val_loss / samples
                        val_losses.append(val_loss)

                        if self.cfg.early_stopping:
                            early_stopping(val_loss)
                            if early_stopping.early_stop:
                                logger.info("Stopped because of EarlyStopping")
                                break

                    if "val_acc" in print_args:
                        val_acc.append(val_correct / val_total)
                    if "val_recall" in print_args:
                        # Do macro averaging recall over all classes
                        recall = torchmetrics.Recall(
                            average="macro", num_classes=self.cfg.model.numclasses
                        ).to(self.cfg.train_args.device)
                        val_recall = recall(outputs, targets).item()
                        val_recalls.append(val_recall)

                    # calculate precision and f1
                    precision = torchmetrics.Precision(
                        average="macro", num_classes=self.cfg.model.numclasses
                    ).to(self.cfg.train_args.device)
                    val_precision = precision(outputs, targets).item()
                    val_precisions.append(val_precision)
                    f1 = torchmetrics.F1Score(
                        average="macro", num_classes=self.cfg.model.numclasses
                    ).to(self.cfg.train_args.device)
                    val_f1 = f1(outputs, targets).item()
                    val_f1s.append(val_f1)

                    # confusion matrix
                    wandb.log(
                        {
                            "val_confusion_matrix": wandb.plot.confusion_matrix(
                                probs=outputs,
                                y_true=targets.cpu().numpy(),
                            )
                        },
                        step=epoch,
                    )

                if ("tst_loss" in print_args) or ("tst_acc" in print_args):
                    samples = 0
                    with torch.no_grad():
                        for _, data in enumerate(testloader):
                            if is_selcon:
                                inputs, targets, _ = data
                            else:
                                inputs, targets = data

                            inputs, targets = inputs.to(
                                self.cfg.train_args.device
                            ), targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            tst_loss += loss.item() * testloader.batch_size
                            samples += targets.shape[0]
                            if "tst_acc" in print_args:
                                if is_selcon:
                                    predicted = outputs
                                else:
                                    _, predicted = outputs.max(1)
                                tst_total += targets.size(0)
                                tst_correct += predicted.eq(targets).sum().item()
                        tst_loss = tst_loss / samples
                        tst_losses.append(tst_loss)

                    if "tst_acc" in print_args:
                        tst_acc.append(tst_correct / tst_total)
                    if "tst_recall" in print_args:
                        # Do macro averaging recall over all classes
                        recall = torchmetrics.Recall(
                            average="macro", num_classes=self.cfg.model.numclasses
                        ).to(self.cfg.train_args.device)
                        tst_recall = recall(outputs, targets).item()
                        tst_recalls.append(tst_recall)

                    # calculate precision and f1
                    precision = torchmetrics.Precision(
                        average="macro", num_classes=self.cfg.model.numclasses
                    ).to(self.cfg.train_args.device)
                    tst_precision = precision(outputs, targets).item()
                    tst_precisions.append(tst_precision)
                    f1 = torchmetrics.F1Score(
                        average="macro", num_classes=self.cfg.model.numclasses
                    ).to(self.cfg.train_args.device)
                    tst_f1 = f1(outputs, targets).item()
                    tst_f1s.append(tst_f1)

                    # confusion matrix
                    wandb.log(
                        {
                            "tst_confusion_matrix": wandb.plot.confusion_matrix(
                                preds=predicted.cpu().numpy(),
                                y_true=targets.cpu().numpy(),
                            )
                        },
                        step=epoch,
                    )

                if "subtrn_acc" in print_args:
                    subtrn_acc.append(subtrn_correct / subtrn_total)

                if "subtrn_losses" in print_args:
                    subtrn_losses.append(subtrn_loss)

                print_str = "Epoch: " + str(epoch + 1)

                """
                ################################################# Results Printing #################################################
                """

                metrics = {}
                for arg in print_args:
                    if arg == "val_loss":
                        print_str += " , " + "Validation Loss: " + str(val_losses[-1])
                        metrics["val_loss"] = val_losses[-1]
                    if arg == "val_acc":
                        print_str += " , " + "Validation Accuracy: " + str(val_acc[-1])
                        metrics["val_acc"] = val_acc[-1]
                    if arg == "val_recall":
                        print_str += (
                            " , " + "Validation Recall: " + str(val_recalls[-1])
                        )
                        metrics["val_recall"] = val_recalls[-1]

                    if arg == "tst_loss":
                        print_str += " , " + "Test Loss: " + str(tst_losses[-1])
                        metrics["tst_loss"] = tst_losses[-1]
                    if arg == "tst_acc":
                        print_str += " , " + "Test Accuracy: " + str(tst_acc[-1])
                        metrics["tst_acc"] = tst_acc[-1]
                    if arg == "tst_recall":
                        print_str += " , " + "Test Recall: " + str(tst_recalls[-1])
                        metrics["tst_recall"] = tst_recalls[-1]

                    # precision and f1
                    print_str
                    if arg == "trn_loss":
                        print_str += " , " + "Training Loss: " + str(trn_losses[-1])
                        metrics["trn_loss"] = trn_losses[-1]
                    if arg == "trn_acc":
                        print_str += " , " + "Training Accuracy: " + str(trn_acc[-1])
                        metrics["trn_acc"] = trn_acc[-1]
                    if arg == "trn_recall":
                        print_str += " , " + "Training Recall: " + str(trn_recalls[-1])
                        metrics["trn_recall"] = trn_recalls[-1]
                    if arg == "subtrn_loss":
                        print_str += " , " + "Subset Loss: " + str(subtrn_losses[-1])
                        metrics["subtrn_loss"] = subtrn_losses[-1]
                    if arg == "subtrn_acc":
                        print_str += " , " + "Subset Accuracy: " + str(subtrn_acc[-1])
                        metrics["subtrn_acc"] = subtrn_acc[-1]
                    if arg == "time":
                        print_str += " , " + "Timing: " + str(timing[-1])
                        metrics["time"] = timing[-1]
                        metrics["cumulative_time"] = total_timing

                # precision and f1
                print_str += " , " + "Training Precision: " + str(trn_precisions[-1])
                metrics["trn_precision"] = trn_precisions[-1]
                print_str += " , " + "Validation Precision: " + str(val_precisions[-1])
                metrics["val_precision"] = val_precisions[-1]
                print_str += " , " + "Test Precision: " + str(tst_precisions[-1])
                metrics["tst_precision"] = tst_precisions[-1]
                print_str += " , " + "Training F1: " + str(trn_f1s[-1])
                metrics["trn_f1"] = trn_f1s[-1]
                print_str += " , " + "Validation F1: " + str(val_f1s[-1])
                metrics["val_f1"] = val_f1s[-1]
                print_str += " , " + "Test F1: " + str(tst_f1s[-1])
                metrics["tst_f1"] = tst_f1s[-1]

                # report metric to ray for hyperparameter optimization
                if (
                    "report_tune" in self.cfg
                    and self.cfg.report_tune
                    and len(dataloader)
                ):
                    tune.report(mean_accuracy=val_acc[-1])

                logger.info(print_str)
                wandb.log(metrics, step=epoch)

            #  report energy every epoch.
            self.report_energy(epoch=epoch)

            """
            ################################################# Checkpoint Saving #################################################
            """

            if ((epoch + 1) % self.cfg.ckpt.save_every == 0) and self.cfg.ckpt.is_save:

                metric_dict = {}

                for arg in print_args:
                    if arg == "val_loss":
                        metric_dict["val_loss"] = val_losses
                    if arg == "val_acc":
                        metric_dict["val_acc"] = val_acc
                    if arg == "tst_loss":
                        metric_dict["tst_loss"] = tst_losses
                    if arg == "tst_acc":
                        metric_dict["tst_acc"] = tst_acc
                    if arg == "trn_loss":
                        metric_dict["trn_loss"] = trn_losses
                    if arg == "trn_acc":
                        metric_dict["trn_acc"] = trn_acc
                    if arg == "subtrn_loss":
                        metric_dict["subtrn_loss"] = subtrn_losses
                    if arg == "subtrn_acc":
                        metric_dict["subtrn_acc"] = subtrn_acc
                    if arg == "time":
                        metric_dict["time"] = timing

                ckpt_state = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": self.loss_function(),
                    "metrics": metric_dict,
                }

                # save checkpoint
                self.save_ckpt(ckpt_state, checkpoint_path)
                logger.info("Model checkpoint saved at epoch: {0:d}".format(epoch + 1))

        """
        ################################################# Results Summary #################################################
        """

        logger.info(
            self.cfg.dss_args.type + " Selection Run---------------------------------"
        )
        logger.info("Final SubsetTrn: {0:f}".format(subtrn_loss))
        if "val_loss" in print_args:
            if "val_acc" in print_args:
                logger.info(
                    "Validation Loss: %.2f , Validation Accuracy: %.2f",
                    val_loss,
                    val_acc[-1],
                )
            else:
                logger.info("Validation Loss: %.2f", val_loss)

        if "tst_loss" in print_args:
            if "tst_acc" in print_args:
                logger.info(
                    "Test Loss: %.2f, Test Accuracy: %.2f", tst_loss, tst_acc[-1]
                )
            else:
                logger.info("Test Data Loss: %f", tst_loss)
        logger.info(
            "---------------------------------------------------------------------"
        )
        logger.info(self.cfg.dss_args.type)
        logger.info(
            "---------------------------------------------------------------------"
        )

        """
        ################################################# Final Results Logging #################################################
        """

        if "val_acc" in print_args:
            val_str = "Validation Accuracy, "
            for val in val_acc:
                val_str = val_str + " , " + str(val)
            logger.info(val_str)

        if "tst_acc" in print_args:
            tst_str = "Test Accuracy, "
            for tst in tst_acc:
                tst_str = tst_str + " , " + str(tst)
            logger.info(tst_str)

        if "time" in print_args:
            time_str = "Time, "
            for t in timing:
                time_str = time_str + " , " + str(t)
            logger.info(timing)

        omp_timing = np.array(timing)
        omp_cum_timing = list(self.generate_cumulative_timing(omp_timing))
        logger.info(
            "Total time taken by %s = %.4f ", self.cfg.dss_args.type, omp_cum_timing[-1]
        )
        wandb.run.summary["total_time_taken"] = omp_cum_timing[-1]
        wandb.run.summary["best_test_accuracy"] = max(tst_acc)
        wandb.run.summary["best_test_f1"] = max(tst_f1s)
        wandb.run.summary["best_test_precision"] = max(tst_precisions)
        wandb.run.summary["best_test_recall"] = max(tst_recalls)
        wandb.finish()
