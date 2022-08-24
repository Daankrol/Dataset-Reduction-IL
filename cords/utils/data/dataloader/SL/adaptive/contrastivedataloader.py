import copy
from dotmap import DotMap
from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import ContrastiveActiveLearningStrategy
import time


class ContrastiveDataLoader(AdaptiveDSSDataLoader):
    """
    Implementation of the ContrastiveDataLoader that serves as the dataloader for the adaptive CAL subset selection strategy.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """

    def __init__(self, train_loader, val_loader, dss_args, logger, *args, **kwargs):
        """
        Constructor function
        """
        assert (
            "model" in dss_args.keys()
        ), "'model' is a compulsory argument for CALSampling. Please provide the model to be used for CAL sampling."
        assert "selection_type" in dss_args.keys(), "'selection_type' is a compulsory argument for CAL. Include it as a key in dss_args"
        assert dss_args.selection_type in ['PerBatch', 'PerClass']
        assert "metric" in dss_args.keys(), "'metric' is a compulsory argument for CAL. Include it as a key in dss_args"
        assert dss_args.metric in ['euclidean', 'cossim'], "'metric' must be either 'euclidean' or 'cossim'"
        super(ContrastiveDataLoader, self).__init__(
            train_loader, val_loader, dss_args, logger, *args, **kwargs
        )
        self.train_model = dss_args.model
        self.strategy = ContrastiveActiveLearningStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model),
                                                          dss_args.loss, dss_args.device, dss_args.num_classes,
                                                          dss_args.linear_layer, dss_args.selection_type, logger, dss_args.metric)

        self.logger.debug("Contrastive dataloader initialized.")

    def _resample_subset_indices(self):
        """
        Function that calls the subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug(
            "Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch)
        )
        self.logger.debug("Selection budget: %d", self.budget)
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict)
        end = time.time()
        self.logger.info(
            "Epoch: {0:d}, CAL subset selection finished, takes {1:.4f}. ".format(
                self.cur_epoch, (end - start)
            )
        )
        return subset_indices, subset_weights
