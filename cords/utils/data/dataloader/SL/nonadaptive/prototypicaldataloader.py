from .nonadaptivedataloader import NonAdaptiveDSSDataLoader
from cords.selectionstrategies.SL import CRAIGStrategy
import time, copy


class PrototypicalDataloaderNonadaptive(NonAdaptiveDSSDataLoader):
    """
    Implements of PrototypicalDataloaderNonadaptive that serves as the dataloader for the nonadaptive Prototypical subset selection strategy from the paper Beyond neural scaling laws.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary required for Prototypical subset selection strategy
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, val_loader, dss_args, logger, *args, **kwargs):
        """
         Constructor function
        """
        assert "num_classes" in dss_args.keys(), "'num_classes' is a compulsory argument for CRAIG. Include it as a key in dss_args"
        
        super(PrototypicalDataloaderNonadaptive, self).__init__(train_loader, val_loader, dss_args,
                                              logger, *args, **kwargs)
        
        self.strategy = CRAIGStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model), dss_args.num_classes, 
                                     dss_args.linear_layer, dss_args.loss, dss_args.device, 
                                     False, dss_args.selection_type, logger, dss_args.optimizer)
        self.train_model = dss_args.model
        self.eta = dss_args.eta
        self.num_cls = dss_args.num_classes
        self.model = dss_args.model
        self.loss = copy.deepcopy(dss_args.loss)
        self.logger.debug('Non-adaptive CRAIG dataloader loader initialized. ')

    def _init_subset_loader(self):
        """
        Function that initializes the subset loader based on the subset indices and the subset weights.
        """
        # All strategies start with random selection
        self.subset_indices, self.subset_weights = self._init_subset_indices()
        self._refresh_subset_loader()

    def _init_subset_indices(self):
        """
        Function that calls the CRAIG strategy for initial subset selection and calculating the initial subset weights.
        """
        start = time.time()
        self.logger.debug('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        end = time.time()
        self.logger.info('Epoch: {0:d}, CRAIG subset selection finished, takes {1:.4f}. '.format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights