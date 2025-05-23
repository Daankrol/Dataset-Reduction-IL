import logging
from abc import abstractmethod
import time
from torch.utils.data import DataLoader
import wandb
from ..dssdataloader import DSSDataLoader
from math import ceil


class AdaptiveDSSDataLoader(DSSDataLoader):
    """
    Implementation of AdaptiveDSSDataLoader class which serves as base class for dataloaders of other
    adaptive subset selection strategies for supervised learning framework.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """

    def __init__(self, train_loader, val_loader, dss_args, logger, *args, **kwargs):

        """
        Constructor function
        """
        super(AdaptiveDSSDataLoader, self).__init__(
            train_loader.dataset, dss_args, logger, *args, **kwargs
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Arguments assertion check
        assert (
            "select_every" in dss_args.keys()
        ), "'select_every' is a compulsory argument. Include it as a key in dss_args"
        assert (
            "device" in dss_args.keys()
        ), "'device' is a compulsory argument. Include it as a key in dss_args"
        assert (
            "kappa" in dss_args.keys()
        ), "'kappa' is a compulsory argument. Include it as a key in dss_args"
        if "inverse_warmup" in dss_args.keys():
            self.inverse_warmup = dss_args["inverse_warmup"]
        else:
            self.inverse_warmup = False
        self.select_every = dss_args.select_every
        self.device = dss_args.device

        if "online" in dss_args.keys() and dss_args.online:
            self.online = True
        else:
            self.online = False

        self.logger.info(
            "Running Data Subset Selection in "
            + ("adaptive" if self.online else "non-adaptive")
            + " mode."
        )
        self.kappa = dss_args.kappa
        if dss_args.kappa > 0:
            assert (
                "num_epochs" in dss_args.keys()
            ), "'num_epochs' is a compulsory argument when warm starting the model(i.e., kappa > 0). Include it as a key in dss_args"
            # self.select_after =  int(dss_args.kappa * dss_args.num_epochs)
            # self.warmup_epochs = ceil(self.select_after * dss_args.fraction)
            self.warmup_epochs = int(dss_args.kappa * dss_args.num_epochs)

            self.logger.info(f"Using {self.warmup_epochs} epochs for warm up.")

        else:
            # self.select_after = 0
            self.warmup_epochs = 0
        self.initialized = False
        self.resampled = False

    def __iter__(self):
        """
        Iter function that returns the iterator of full data loader or data subset loader or empty loader based on the
        warmstart kappa value.
        """
        if self.online:
            if not self.inverse_warmup:
                # Use normal warmup where warmup is training with all data and then switching to subset
                self.initialized = True
                if self.cur_epoch < self.warmup_epochs + 1:
                    self.logger.info(
                        f"Epoch: {self.cur_epoch} using full dataset for warm-start."
                    )
                    loader = self.wtdataloader
                    self.resampled = False
                # elif self.cur_epoch == self.warmup_epochs + 1:
                #     self.logger.info(
                #         f"Epoch: {self.cur_epoch} finished with warm up so forcing a resample."
                #     )
                #     self.resample()
                #     loader = self.subset_loader
                else:
                    self.logger.info(
                        f"Epoch: {self.cur_epoch}, reading dataloader... {self.cur_epoch, self.select_every}"
                    )
                    if (
                        self.cur_epoch - self.warmup_epochs - 1
                    ) % self.select_every == 0:
                        self.resample()
                    else:
                        self.resampled = False
                    # if (
                    #     (self.cur_epoch + self.warmup_epochs) % self.select_every == 0
                    # ) and (self.cur_epoch > 1):
                    #     self.resample()
                    # else:
                    #     self.resampled = False
                    loader = self.subset_loader
                    # print("Size: ", len(loader))
                    self.logger.info(
                        "Epoch: {0:d}, finished reading dataloader. ".format(
                            self.cur_epoch
                        )
                    )
            else:
                # Use inverse warmup where warmup is training with subset and then switching to full
                if self.cur_epoch < self.warmup_epochs + 1:
                    self.logger.info(
                        "Epoch: {0:d}, inverse warm-up, reading dataloader...".format(
                            self.cur_epoch, self.warmup_epochs
                        )
                    )
                    if (self.cur_epoch - 1) % self.select_every == 0:
                        self.resample()
                    else:
                        self.resampled = False
                    loader = self.subset_loader
                else:
                    self.logger.info(
                        "Epoch: inverse warm-up done, using full dataset.".format(
                            self.cur_epoch, self.warmup_epochs
                        )
                    )
                    loader = self.wtdataloader
                    self.resampled = False

        else:
            if not self.initialized:
                self.resample()
                self.initialized = True
                loader = self.subset_loader
            else:
                loader = self.subset_loader

        self.cur_epoch += 1
        return loader.__iter__()
        #
        # if self.warmup_epochs < self.cur_epoch <= self.select_after:
        #     self.logger.debug(
        #         "Skipping epoch {0:d} due to warm-start option. ".format(self.cur_epoch, self.warmup_epochs))
        #     loader = DataLoader([])
        #
        # elif self.cur_epoch <= self.warmup_epochs:
        #     self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
        #     loader = self.wtdataloader
        #     self.logger.debug('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
        # else:
        #     self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
        #     if ((self.cur_epoch - 1) % self.select_every == 0) and (self.cur_epoch > 1):
        #         self.resample()
        #     loader = self.subset_loader
        #     self.logger.debug('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
        #
        # self.cur_epoch += 1
        # return loader.__iter__()

    def __len__(self) -> int:
        """
        Returns the length of the current data loader
        """
        if self.online:
            if self.cur_epoch < self.warmup_epochs + 1:
                return (
                    len(self.wtdataloader)
                    if not self.inverse_warmup
                    else len(self.subset_loader)
                )
            else:
                return (
                    len(self.subset_loader)
                    if not self.inverse_warmup
                    else len(self.wtdataloader)
                )
        else:
            return len(self.subset_loader)
        #
        # if self.warmup_epochs < self.cur_epoch <= self.select_after:
        #     self.logger.debug(
        #         "Skipping epoch {0:d} due to warm-start option. ".format(self.cur_epoch, self.warmup_epochs))
        #     loader = DataLoader([])
        #     return len(loader)
        #
        # elif self.cur_epoch <= self.warmup_epochs:
        #     self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
        #     loader = self.wtdataloader
        #     #self.logger.debug('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
        #     return len(loader)
        # else:
        #     self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
        #     loader = self.subset_loader
        #     return len(loader)

    def resample(self):
        """
        Function that resamples the subset indices and recalculates the subset weights
        """
        self.resampled = True
        # start timing
        start = time.time()
        self.subset_indices, self.subset_weights = self._resample_subset_indices()
        # end timing
        end = time.time()
        wandb.log({"resample_duration": end - start})
        self.logger.debug("Subset indices length: %d", len(self.subset_indices))
        self._refresh_subset_loader()
        self.logger.debug(
            "Subset loader initiated, args: %s, kwargs: %s",
            self.loader_args,
            self.loader_kwargs,
        )
        self.logger.debug(
            "Subset selection finished, Training data size: %d, Subset size: %d",
            self.len_full,
            len(self.subset_loader.dataset),
        )

    @abstractmethod
    def _resample_subset_indices(self):
        """
        Abstract function that needs to be implemented in the child classes.
        Needs implementation of subset selection implemented in child classes.
        """
        raise Exception("Not implemented.")
