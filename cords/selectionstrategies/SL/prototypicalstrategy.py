from .dataselectionstrategy import DataSelectionStrategy
import time
import torch
import torch.nn.functional as F
import numpy as np


# Adapted from: https://github.com/PatrickZH/DeepCore/blob/main/deepcore/methods/uncertainty.py 

class PrototypicalStrategy(DataSelectionStrategy):
    def __init__(
        self,
        trainloader,
        valloader,
        model,
        num_classes,
        linear_layer,
        loss,
        device,
        selection_type,
        pretrained_model,
        logger
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
        self.pretrained_model = pretrained_model
        self.N_trn = len(trainloader.sampler.data_source)

    def select(self, budget, model_params):
        start_time = time.time()
        self.logger.info(f"Started Prototypical selection.")
        self.logger.info("Budget: {0:d}".format(budget))
        self.fraction = budget / self.N_trn
        
        # per-class sampling
        self.get_labels()
        indices = np.array([], dtype=np.int64)
        scores = []
        # we want to fill the budget with samples from each class
        for c in range(self.num_classes):
            class_index = np.arange(self.N_trn)[self.trn_lbls == c]
            budget_for_class = int(self.fraction * len(class_index))

            indices = np.append(indices, self.select_from_class(class_index, budget_for_class))

        end_time = time.time()
        self.logger.info(
            "Prototypical algorithm Subset Selection time is: {0:.4f}.".format(
                end_time - start_time
            )
        )
        self.logger.info("Selected {} samples with a budget of {}".format(len(indices), budget))

        return indices, torch.ones(len(indices))

    def select_from_class(self, class_indices, budget_for_class):
        self.pretrained_model.eval()
        # compute the mean feature vector for this class
        mean_feature = torch.zeros(self.pretrained_model.embDim).to(self.device)
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.trainloader.dataset, class_indices), batch_size=32, shuffle=False)
        for x, y in loader:
            x = x.to(self.device)
            _, e = self.pretrained_model(x, last=True, freeze=True)
            mean_feature += e.sum(dim=0)
        mean_feature /= len(class_indices)

        # for each sample in the class, compute the (euclidian) distance to the mean feature vector
        # select the top 'budget_for_class' samples with the highest distance
        distances = []
        for i in class_indices:
            x = self.trainloader.dataset[i][0].unsqueeze(0).to(self.device)
            _, e = self.pretrained_model(x, last=True, freeze=True)
            distances.append(F.pairwise_distance(e, mean_feature.unsqueeze(0)).item())
        distances = np.array(distances)
        indices = class_indices[np.argsort(distances)[-budget_for_class:]]
        return indices


