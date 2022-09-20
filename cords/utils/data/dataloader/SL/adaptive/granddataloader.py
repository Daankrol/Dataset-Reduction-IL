import copy

from dotmap import DotMap

from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import GrandStrategy
import time


class GrandDataLoader(AdaptiveDSSDataLoader):
    """
    Implementation of the GrandDataLoader that serves as the dataloader for the adaptive Grand subset selection strategy.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    dss_args: dict
        Data subset selection arguments dictionary required for Random subset selection strategy
    logger: class
        Logger for logging the information
    """

    def __init__(self, train_loader, val_loader, dss_args, logger, *args, **kwargs):
        """
        Constructor function
        """
        assert (
            "model" in dss_args.keys()
        ), "'model' is a compulsory argument for GrandSampling. Please provide the model to be used for Grand sampling."
        assert dss_args.selection_type in ['Supervised', 'PerClass'], "selection_type must be either 'Supervised' or 'PerClass'"
        super(GrandDataLoader, self).__init__(
            train_loader, val_loader, dss_args, logger, *args, **kwargs
        )
        if "repeats" not in dss_args.keys():
            dss_args.repeats = 10
        self.strategy = GrandStrategy(
            train_loader, val_loader, copy.deepcopy(dss_args.model),
            dss_args.num_classes, dss_args.linear_layer, dss_args.loss,
            dss_args.device, dss_args.selection_type, logger,
            repeats=dss_args.repeats )

        self.logger.debug("Grand dataloader initialized.")

    def _resample_subset_indices(self):
        """
        Function that calls the Grand subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug(
            "Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch)
        )
        self.logger.debug("Grand budget: %d", self.budget)
        subset_indices, subset_weights = self.strategy.select(self.budget, None)
        end = time.time()
        self.logger.info(
            "Epoch: {0:d}, Grand subset selection finished, takes {1:.4f}. ".format(
                self.cur_epoch, (end - start)
            )
        )
        return subset_indices, subset_weights
