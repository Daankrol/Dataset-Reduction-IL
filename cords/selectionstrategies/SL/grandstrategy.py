from .dataselectionstrategy import DataSelectionStrategy
import time
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from cords.utils.models.efficientnet import EfficientNetB0_PyTorch
from torch.utils.data import DataLoader


class GrandStrategy(DataSelectionStrategy):
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
            logger,
            repeats=10,
            train_epochs=0,
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
        # cross entropy loss
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterion.__init__()
        self.selection_type = selection_type
        self.repeats = repeats
        self.train_epochs = train_epochs

    def select(self, budget, model_params):
        # for 10 runs do:
        #   initialize a non-pretrained model with random weights
        #   calculate the GraND scores for all samples by the normed gradients
        # average the scores over the 10 runs
        # larger (norm) scores are more important and should be selected

        norm_matrix = torch.zeros((self.N_trn, 10), requires_grad=False).to(self.device)
        self.fraction = budget / self.N_trn

        # use a trainloader without shuffling such that the indices for PerClass selection are correct
        dataloader = DataLoader(self.trainloader.dataset, batch_size=self.trainloader.batch_size, shuffle=False,
                                pin_memory=self.trainloader.pin_memory, num_workers=self.trainloader.num_workers)

        for run in range(self.repeats):
            self.logger.debug('GRAND: model run {}/{}'.format(run + 1, self.repeats))
            random_seed = int(time.time() * 1000) % 100000
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

            # initialize a non-pretrained model with random weights
            model = EfficientNetB0_PyTorch(num_classes=self.num_classes, pretrained=False, fine_tune=True).to(
                self.device)
            model.embedding_recorder.record_embedding = True
            embedding_dim = model.get_embedding_dim()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            # get metric at initialization time or train for some epochs first
            if self.train_epochs > 0:
                self.train(model, optimizer, self.train_epochs)

            model.eval()

            # calculate the GraND scores for all samples by the normed gradients
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(F.softmax(outputs.requires_grad_(True), dim=1), targets).sum()
                batch_num = inputs.size(0)

                with torch.no_grad():
                    bias_param_grads = torch.autograd.grad(loss, outputs)[0]
                    grads = torch.cat([bias_param_grads, (
                            model.embedding_recorder.embedding.view(batch_num, 1, embedding_dim).repeat(1,
                                                                                                        self.num_classes,
                                                                                                        1) * bias_param_grads.view(
                        batch_num, self.num_classes, 1).repeat(1, 1, embedding_dim)).
                                      view(batch_num, -1)], dim=1)
                    norm_matrix[
                    batch_idx * self.trainloader.batch_size:min((batch_idx + 1) * self.trainloader.batch_size,
                                                                self.N_trn),
                    run] = torch.norm(grads, dim=1, p=2)

        # average the scores over the 10 runs
        norm_matrix = torch.mean(norm_matrix, dim=1).cpu().detach().numpy()
        if self.selection_type == 'Supervised':
            # select the samples with the highest scores
            selected_idxs = np.argsort(norm_matrix)[-budget:]
        elif self.selection_type == 'PerClass':
            labels = self.get_labels_of_dataloader(dataloader)
            selected_idxs = np.array([], dtype=np.int64)
            for i in range(self.num_classes):
                class_idxs = np.arange(self.N_trn)[labels == i]
                class_budget = round(self.fraction * len(class_idxs))
                selected_idxs = np.concatenate(
                    (selected_idxs, class_idxs[np.argsort(norm_matrix[class_idxs])[-class_budget:]]))

        return selected_idxs, torch.ones(len(selected_idxs))

    def train(self, model, optimizer, epochs=10):
        model.train()
        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                optimizer.zero_grad()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
