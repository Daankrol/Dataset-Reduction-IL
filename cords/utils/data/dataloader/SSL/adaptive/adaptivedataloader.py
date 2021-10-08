import logging
from abc import abstractmethod
from torch.utils.data import DataLoader
from ..dssdataloader import DSSDataLoader
from cords.utils.data.datasets.SSL.utils import InfiniteSampler

class AdaptiveDSSDataLoader(DSSDataLoader):
    def __init__(self, train_loader, val_loader, dss_args, verbose=False, *args,
                 **kwargs):
        super(AdaptiveDSSDataLoader, self).__init__(train_loader.dataset, dss_args,
                                                    verbose=verbose, *args, **kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        """
         Arguments assertion check
        """
        assert "select_every" in dss_args.keys(), "'select_every' is a compulsory argument. Include it as a key in dss_args"
        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        assert "kappa" in dss_args.keys(), "'kappa' is a compulsory argument. Include it as a key in dss_args"
        assert "batch_size" in kwargs.keys(), "'batch_size' is a compulsory argument. Include it as a key in kwargs"
        assert "sampler" not in kwargs.keys(), "'sampler' is a prohibited argument. Do not include it as a key in kwargs"
        self.select_every = dss_args.select_every
        self.sel_iteration = int((self.select_every * len(self.dataset) * dss_args.fraction) // (kwargs.batch_size))  
        self.device = dss_args.device
        self.kappa = dss_args.kappa
        # kappa_iterations = int()
        if dss_args.kappa > 0:
            assert "num_iters" in dss_args.keys(), "'num_iters' is a compulsory argument when warm starting the model(i.e., kappa > 0). Include it as a key in dss_args"
            self.select_after =  int(self.kappa * self.num_iters * self.fraction)
        else:
            self.select_after = 0
        self.wtdataloader = DataLoader(self.wt_trainset,
                                       sampler=InfiniteSampler(len(self.wt_trainset), self.select_after * kwargs.batch_size),
                                       *self.loader_args, **self.loader_kwargs)
        self.initialized = False
        
    
    def __iter__(self):
        self.initialized = True
        if self.cur_iter <= self.select_after:
            if self.verbose:
                logging.info('Iteration: {0:d}, reading dataloader... '.format(self.cur_iter))
            loader = self.wtdataloader
        else:
            if self.cur_iter > 1:
                self.resample()
            loader = self.subset_loader
            if self.verbose:
                logging.info('Iteration: {0:d}, finished reading dataloader. '.format(self.cur_iter))
        self.cur_iter += len(list(loader.batch_sampler))
        return loader.__iter__()

    def resample(self):
        self.subset_indices, self.subset_weights = self._resample_subset_indices()
        logging.debug("Subset indices length: %d", len(self.subset_indices))
        self._refresh_subset_loader()
        logging.debug("Subset loader initiated, args: %s, kwargs: %s", self.loader_args, self.loader_kwargs)
        logging.info('Subset selection finished, Training data size: %d, Subset size: %d',
                     self.len_full, len(self.subset_loader.dataset))

    @abstractmethod
    def _resample_subset_indices(self):
        raise Exception('Not implemented. ')