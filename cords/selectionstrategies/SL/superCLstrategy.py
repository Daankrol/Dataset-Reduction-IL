import numpy as np
import torch
import torch.nn.functional as F
from .dataselectionstrategy import DataSelectionStrategy
import time
from cords.utils.models.efficientnet import EfficientNetB0_PyTorch

class SupervisedContrastiveLearningStrategy(DataSelectionStrategy):
    """
    This class extends :class:`selectionstrategises.supervisedlearning.dataselectionstrategy.DataSelectionStrategy`
    to use Supervised Contrastive Learning (Super-CL) as a data selection strategy.

    Parameters
    ----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss_type: class
        The type of loss criterion
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    selection_type: str
        PerClass or PerBatch
    """

    def __init__(self, trainloader, valloader, model, loss,
                 device, num_classes, selection_type, logger, k=10, weighted=True):
        super().__init__(trainloader, valloader, model, num_classes,None, loss, device, logger)
        self.selection_type = selection_type
        self.weighted = weighted
        self.k = k
        self.knn = None
        self.pretrained_model = EfficientNetB0_PyTorch(num_classes=self.num_classes, pretrained=True, fine_tune=False).to(self.device)
        # disable gradients for the pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.eval()
    

    @torch.no_grad()
    def find_knn(self):
        "Find k-nearest-neighbour datapoints based on label info and feature embedding by a pretrained network"
        if self.knn is not None:
            return
        self.logger.info('Finding k-nearest-neighbours')
        self.get_labels()

        # for each sample in a class, compute distance in feature space to all other sample in the same class
        # and find the k-nearest-neighbours
        knn = torch.zeros((self.N_trn, self.k), dtype=torch.int32)
        for c in range(self.num_classes):
            class_indices = np.where(self.trn_lbls == c)[0]
            embeddings = torch.zeros((len(class_indices), self.pretrained_model.embDim)).to(self.device)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(self.trainloader.dataset, class_indices),
                batch_size=self.trainloader.batch_size, 
                shuffle=False, num_workers=self.trainloader.num_workers)
            for i, (inputs, _) in enumerate(loader):
                inputs = inputs.to(self.device)
                embeddings[i * self.trainloader.batch_size:(i + 1) * self.trainloader.batch_size] = self.pretrained_model(inputs, last=True, freeze=True).detach()
            
            # calculate pairwise distances and for each sample, find the k-nearest-neighbours, add their indices sorted by their distance
            dist = torch.nn.PairwiseDistance(p=2)(embeddings, embeddings)
            dist = dist.cpu().numpy()
            for i in range(len(class_indices)):
                knn[class_indices[i]] = torch.from_numpy(np.argsort(dist[i])[1:self.k+1])

        del self.pretrained_model  # only need this at the start. 
        self.knn = knn

    @torch.no_grad()
    def calculate_divergence(self):
        # We use the current training model to determine the probabilities 
        self.logger.info('Calculating divergence')
        self.model.eval()

        loader = torch.utils.data.DataLoader(self.trainloader.dataset, batch_size=self.trainloader.batch_size, num_workers=self.trainloader.num_workers)
        probs = torch.zeros([self.N_trn, self.num_classes]).to(self.device)
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs, freeze=True)
            probs[i * self.trainloader.batch_size:(i + 1) * self.trainloader.batch_size] = F.softmax(outputs, dim=1)

        # Calculate KL-divergence between sample PDF and all neighbours of the same class.
        # Average the divergence scores over all neighbours.
        divergence_scores = torch.zeros(self.N_trn).to(self.device)
        for i in range(0, self.N_trn, loader.batch_size):
            # calculate divergence for a batch of samples
            sample_probs = probs[i:i+loader.batch_size]
            sample_probs = sample_probs.unsqueeze(1).repeat(1, self.k, 1)
            sample_probs = sample_probs.reshape(-1, self.num_classes)
            sample_neigbour_probs = probs[self.knn[i:i+loader.batch_size].reshape(-1)]
            divergence = F.kl_div(sample_probs.log(), sample_neigbour_probs, reduction='none').sum(dim=1)
            divergence = divergence.reshape(loader.batch_size, self.k)
            divergence_scores[i:i+loader.batch_size] = divergence.mean(dim=1)
        self.divergence_scores = divergence_scores.cpu().numpy()


    def select(self, budget, model_params):
        start_time = time.time()
        self.logger.info(f'Started {self.selection_type} CAL selection.')
        self.update_model(model_params)
        self.find_knn()
        self.fraction = budget / self.N_trn
        self.calculate_divergence()

        # Higher score means closer to decision boundary and thus more important
        if self.selection_type == 'PerClass':
            indices = np.array([], dtype=np.int32)
            for c in range(self.num_classes):
                class_indices = torch.arange(self.N_trn)[self.trn_lbls == c]
                scores = self.divergence_scores[class_indices]
                num_samples = int(self.fraction * len(class_indices))
                # add the indices of the selected samples that have the highest KL divergence
                indices = np.concatenate((indices, class_indices[torch.argsort(scores, descending=True)[:num_samples]].cpu().numpy()))
        else:
            indices = torch.argsort(self.divergence_scores, descending=True)[:int(self.fraction * self.N_trn)].cpu().numpy()

        end_time = time.time()
        self.logger.info(f'Super-CL algorithm took {end_time - start_time} seconds.')
        self.logger.info('Selected {}  samples with a budget of {}'.format(len(indices), budget))

        if self.weighted:
            weights = self.divergence_scores[indices]
            return indices, weights

        return indices, torch.ones(len(indices))
