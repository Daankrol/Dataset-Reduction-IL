import copy
from cords.selectionstrategies.SL.superCLstrategy import SupervisedContrastiveLearningStrategy
from .adaptivedataloader import AdaptiveDSSDataLoader
import time


class SupervisedContrastiveLearningDataLoader(AdaptiveDSSDataLoader):
    """
    Implementation of the Supervised Contrastive DataLoader that serves as the dataloader for the adaptive super-CL subset selection strategy.

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
        ), "'model' is a compulsory argument for CALSampling. Please provide the model to be used for Super-CL sampling."
        assert "selection_type" in dss_args.keys(), "'selection_type' is a compulsory argument for Super-CL. Include it as a key in dss_args"
        assert dss_args.selection_type in ['PerBatch', 'PerClass'], "selection_type should be either PerBatch or PerClass"
        if "k" not in dss_args.keys():
            dss_args.k = 10
        if "weighted" not in dss_args.keys():
            dss_args.weighted = True
        super(SupervisedContrastiveLearningDataLoader, self).__init__(
            train_loader, val_loader, dss_args, logger, *args, **kwargs
        )
        self.train_model = dss_args.model
        self.strategy = SupervisedContrastiveLearningStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model),
                                                          dss_args.loss, dss_args.device, dss_args.num_classes,
                                                          dss_args.selection_type, logger,dss_args.k, dss_args.weighted)

        self.logger.debug("Supervised Contrastive Learning dataloader initialized.")
        self.logger.debug(f"Using {dss_args.selection_type}, weights based on divergence: {dss_args.weighted}")

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
            "Epoch: {0:d}, Supervised Contrastive subset selection finished, takes {1:.4f}. ".format(
                self.cur_epoch, (end - start)
            )
        )
        return subset_indices, subset_weights
