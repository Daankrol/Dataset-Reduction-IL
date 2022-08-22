import copy

from dotmap import DotMap

from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import UncertaintyStrategy, SubmodularSelectionStrategy
import time


class SubmodularDataloader(AdaptiveDSSDataLoader):
    """
    Implementation of the SubmodularDataloaders that serves as the dataloader for the adaptive Submodular subset selection strategy.

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
        assert (
            "model" in dss_args.keys()
        ), "'model' is a compulsory argument for Submodular. Please provide the model to be used for uncertainty sampling."
        assert dss_args.selection_type in ['PerClass', 'Supervised', 'PerBatch']
        assert dss_args.submod_func_type in ['facility-location', 'graph-cut', 'sum-redundancy', 'saturated-coverage']
        super(SubmodularDataloader, self).__init__(
            train_loader, val_loader, dss_args, logger, *args, **kwargs
        )
        self.train_model = dss_args.model
        self.strategy = SubmodularSelectionStrategy(
            train_loader, val_loader, copy.deepcopy(dss_args.model), dss_args.loss,
            dss_args.device, dss_args.num_classes,
            dss_args.linear_layer,
            dss_args.if_convex,
            dss_args.selection_type,
            dss_args.submod_func_type,
            dss_args.optimizer, # two-stage is default.
            # 'random' : randomly select elements (dummy optimizer)
            # 'modular' : approximate the function using its modular upper bound
            # 'naive' : the naive greedy algorithm
            # 'lazy' : the lazy (or accelerated) greedy algorithm
            # 'approximate-lazy' : the approximate lazy greedy algorithm
            # 'two-stage' : starts with naive and switches to lazy
            # 'stochastic' : the stochastic greedy algorithm
            # 'sample' : randomly take a subset and perform selection on that
            # 'greedi' : the GreeDi distributed algorithm
            # 'bidirectional' : the bidirectional greedy algorithm
        )

        self.logger.debug("Submodular dataloader initialized.")

    def _resample_subset_indices(self):
        """
        Function that calls the Random subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug(
            "Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch)
        )
        self.logger.debug("Uncertainty budget: %d", self.budget)
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict)
        end = time.time()
        self.logger.info(
            "Epoch: {0:d}, Submodular subset selection finished, takes {1:.4f}. ".format(
                self.cur_epoch, (end - start)
            )
        )
        return subset_indices, subset_weights
