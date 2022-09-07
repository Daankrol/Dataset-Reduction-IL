import copy

from dotmap import DotMap

from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import PrototypicalStrategy
import time


class PrototypicalDataLoader(AdaptiveDSSDataLoader):
    """
    Implementation of the PrototypicalDataLoader that serves as the dataloader for the adaptive Prototypical subset selection strategy.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    dss_args: dict
        Data subset selection arguments dictionary required for Prototypicalsubset selection strategy
    logger: class
        Logger for logging the information
    """

    def __init__(self, train_loader, val_loader, dss_args, logger, *args, **kwargs):
        """
        Constructor function
        """
        assert (
            "model" in dss_args.keys()
        ), "'model' is a compulsory argument for PrototypicalDataLoader. Please provide the model to be used for Prototypical sampling."
        super(PrototypicalDataLoader, self).__init__(
            train_loader, val_loader, dss_args, logger, *args, **kwargs
        )
        self.train_model = dss_args.model
        self.strategy = PrototypicalStrategy(
            train_loader, val_loader, copy.deepcopy(dss_args.model),
            dss_args.num_classes, dss_args.linear_layer, dss_args.loss,
            dss_args.device, dss_args.selection_type, dss_args.pretrained_model, logger)

        self.logger.debug("Prototypical dataloader initialized.")

    def _resample_subset_indices(self):
        """
        Function that calls the Prototypical subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug(
            "Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch)
        )
        self.logger.debug("Prototypical budget: %d", self.budget)
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict)
        end = time.time()
        self.logger.info(
            "Epoch: {0:d}, Prototypical subset selection finished, takes {1:.4f}. ".format(
                self.cur_epoch, (end - start)
            )
        )
        return subset_indices, subset_weights
