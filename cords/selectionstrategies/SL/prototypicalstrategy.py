from .dataselectionstrategy import DataSelectionStrategy
import time
import torch
import torch.nn.functional as F
import numpy as np
from cords.utils.models.efficientnet import EfficientNetB0_PyTorch

class PrototypicalStrategy(DataSelectionStrategy):
    def __init__(
        self,
        trainloader,
        valloader,
        num_classes,
        linear_layer,
        loss,
        device,
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


    def select(self, budget, model_params):
        self.pretrained_model = EfficientNetB0_PyTorch(num_classes=self.num_classes, pretrained=True, fine_tune=False).to(self.device)
        self.pretrained_model.eval()
        start_time = time.time()
        self.logger.info(f"Started Prototypical selection.")
        self.logger.info("Budget: {0:d}".format(budget))
        self.fraction = budget / self.N_trn
        
        # per-class sampling
        self.get_labels()
        indices = np.array([], dtype=np.int64)

        # we want to fill the budget with samples from each class
        for c in range(self.num_classes):
            self.logger.debug(f'Computing prototype and selecting samples for class {c}')
            class_index = np.arange(self.N_trn)[self.trn_lbls == c]
            budget_for_class = int(self.fraction * len(class_index))
            indices = np.append(indices, self.select_from_class(class_index, budget_for_class))
            self.logger.debug(f'Selected {len(indices)} samples for class {c} with a class-budget of {budget_for_class}')

        end_time = time.time()
        self.logger.info(
            "Prototypical algorithm Subset Selection time is: {0:.4f}.".format(
                end_time - start_time
            )
        )
        self.logger.info("Selected {} samples with a budget of {}".format(len(indices), budget))
        self.logger.debug("Selected {} unique samples from {} total samples".format(len(np.unique(indices)), len(indices)))

        # remove the model from cuda memory 
        del self.pretrained_model
        torch.cuda.empty_cache()

        return indices, [1 for _ in range(len(indices))]

    @torch.no_grad()
    def select_from_class(self, class_indices, budget_for_class):
        # compute the mean feature vector for this class
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.trainloader.dataset, class_indices), batch_size=32, shuffle=False)
        # with torch.no_grad():
        mean_feature = torch.zeros(self.pretrained_model.embDim).to(self.device)
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


