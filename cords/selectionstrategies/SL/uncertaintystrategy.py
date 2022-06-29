from .dataselectionstrategy import DataSelectionStrategy
import time
import torch
import torch.nn.functional as F
import numpy as np

class UncertaintyStrategy(DataSelectionStrategy):
    def __init__(
        self,
        trainloader,
        valloader,
        model,
        num_classes,
        linear_layer,
        loss,
        device,
        logger,
    ):
        super().__init__(
            trainloader,
            valloader,
            model,
            num_classes,
            linear_layer,
            loss,
            device,
            logger,
        )
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.N_trn = len(trainloader.sampler.data_source)

    def select(self, budget, model_params):
        start_time = time.time()
        self.model.load_state_dict(model_params)
        self.logger.info("Started uncertainty selection")
        self.logger.info("Budget: {0:d}".format(budget))
        idxs = []
        uncertainties = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.trainloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                out = self.model(inputs)
                # turn this into probability
                out = F.softmax(out, dim=1)

                # out is of shape (batch_size, num_classes)
                # get the probability for the target classess for each sample in the batch
                # this is of shape (batch_size)
                probs = out[range(out.shape[0]), targets]
                # compute the uncertainty of the model on each sample in the batch
                # this is of shape (batch_size)
                uncertainty = 1 - probs
                # extend the uncertainty vector with the uncertainties of the batch
                uncertainties.extend(uncertainty.cpu().numpy())
                # add the idx of each sample with respect to the full dataset to the list of idxs
                idxs.extend(i * np.arange(0, uncertainty.shape[0]))

        # sort the idxs by descending uncertainty
        idxs = np.array(idxs)
        uncertainties = np.array(uncertainties)
        idxs = idxs[np.argsort(uncertainties)[::-1]]
        uncertainties = uncertainties[np.argsort(uncertainties)[::-1]]

        # select the top k samples where k == budget
        # return the indices of the selected samples

        idxs = idxs[:budget]
        gammas = torch.ones(len(idxs))
        end_time = time.time()
        self.logger.info(
            "Uncertainty algorithm Subset Selection time is: {0:.4f}".format(
                end_time - start_time
            )
        )
        return idxs, gammas
