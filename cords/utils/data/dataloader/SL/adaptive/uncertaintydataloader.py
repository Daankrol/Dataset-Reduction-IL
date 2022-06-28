from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import RandomStrategy
import time


class UncertaintyDataLoader(AdaptiveDSSDataLoader):
    """
    Implementation of the UncertaintyDataLoader that serves as the dataloader for the adaptive Uncertainty subset selection strategy.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    dss_args: dict
        Data subset selection arguments dictionary required for Random subset selection strategy
    logger: class
        Logger for logging the information
    """

    def __init__(self, train_loader, dss_args, logger, *args, **kwargs):
        """
        Constructor function
        """
        assert (
            "model" in dss_args.keys()
        ), "'model' is a compulsory argument for UncertaintySampling. Please provide the model to be used for uncertainty sampling."

        super(RandomDataLoader, self).__init__(
            train_loader, train_loader, dss_args, logger, *args, **kwargs
        )
        self.strategy = UncertaintyStrategy(train_loader, online=False)

        UncertaintyStrategy(train_loader, online=False)
        self.logger.debug("Random dataloader initialized.")

    def _resample_subset_indices(self):
        """
        Function that calls the Random subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug(
            "Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch)
        )
        self.logger.debug("Random budget: %d", self.budget)
        subset_indices, subset_weights = self.strategy.select(self.budget)
        end = time.time()
        self.logger.info(
            "Epoch: {0:d}, Random subset selection finished, takes {1:.4f}. ".format(
                self.cur_epoch, (end - start)
            )
        )
        return subset_indices, subset_weights
