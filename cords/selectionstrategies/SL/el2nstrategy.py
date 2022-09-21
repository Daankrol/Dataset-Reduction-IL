from .dataselectionstrategy import DataSelectionStrategy
import time
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from cords.utils.models.efficientnet import EfficientNetB0_PyTorch
from torch.utils.data import DataLoader

class EL2NStrategy(DataSelectionStrategy):
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
        train_epochs=20,
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
        #   train for <train_epochs> epochs
        #   calculate the EL2n scores for all samples
        #       el2n scores are  E||p(w,x) - y||^2
        #       where p(w,x) is the predicted probability of class w for sample x
        #       and y is the true label in one-hot encoding
        # average the scores over the 10 runs
        # larger (norm) scores are more important and should be selected

        norm_matrix = torch.zeros((self.N_trn, self.repeats), requires_grad=False).to(self.device)
        self.fraction = budget / self.N_trn


        dataloader = DataLoader(self.trainloader.dataset, batch_size=self.trainloader.batch_size, shuffle=False)
        
        for run in range(self.repeats):
            self.logger.debug('EL2N: model run {}/{}'.format(run+1, self.repeats))
            random_seed = int(time.time() * 1000) % 100000
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

            # initialize a non-pretrained model with random weights
            model = EfficientNetB0_PyTorch(num_classes=self.num_classes, pretrained=False, fine_tune=True).to(self.device)
            model.embedding_recorder.record_embedding = True

            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            # get metric at initialization time or train for some epochs first
            if self.train_epochs > 0:
                self.train(model, optimizer, self.train_epochs)
        
            model.eval()

            # calculate the EL2n scores for all samples
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    optimizer.zero_grad()
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    outputs = F.softmax(outputs, dim=1)
                    # make sure y (targets) is one-hot encoded
                    y = torch.zeros(outputs.shape).to(self.device)
                    y[torch.arange(y.shape[0]), targets] = 1
                    # calculate the L2 norm
                    l2_norm = torch.norm(outputs - y, p=2, dim=1)
                    cur_batch_size = inputs.shape[0]
                    norm_matrix[batch_idx * self.trainloader.batch_size : batch_idx * self.trainloader.batch_size + cur_batch_size, run] = l2_norm

        # average the scores over the runs
        norm_matrix = torch.mean(norm_matrix, dim=1).cpu().detach().numpy()
        # select the samples with the highest scores
        if self.selection_type == 'Supervised':
            selected_idxs = np.argsort(norm_matrix)[-budget:]
        elif self.selection_type == 'PerClass':
            labels = self.get_labels_of_dataloader(dataloader)
            selected_idxs = np.array([], dtype=np.int64)
            for i in range(self.num_classes):
                class_idxs = np.arange(self.N_trn)[labels == i]
                class_budget = round(self.fraction * len(class_idxs))
                selected_idxs = np.concatenate((selected_idxs, class_idxs[np.argsort(norm_matrix[class_idxs])[-class_budget:]]))
            
        return selected_idxs, torch.ones(len(selected_idxs))

    def train(self, model, optimizer, epochs=10):
        ## Note: here we can (and should) use data shuffling so just use the trainloader. 
        model.train()
        self.logger.info('Training EL2N model for {} epochs'.format(epochs))
        for epoch in range(epochs):
            self.logger.debug('EL2N: epoch {}/{}'.format(epoch+1, epochs))
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                optimizer.zero_grad()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()



