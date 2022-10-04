import numpy as np
import torch
import torch.nn.functional as F
from .dataselectionstrategy import DataSelectionStrategy
from sklearn.metrics import pairwise_distances
import time
from cords.utils.models.efficientnet import EfficientNetB0_PyTorch
import pickle 
import os


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
        super().__init__(trainloader, valloader, model, num_classes, None, loss, device, logger)
        self.selection_type = selection_type
        self.weighted = weighted
        self.k = k
        self.knn = None
        self.pretrained_model = EfficientNetB0_PyTorch(num_classes=self.num_classes, pretrained=True,
                                                       fine_tune=False).to(self.device)
        # disable gradients for the pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.eval()

    @torch.no_grad()
    def find_knn(self):
        "Find k-nearest-neighbour datapoints based on label info and feature embedding by a pretrained network"
        if self.knn is not None:
            return
        # load with pickle if file is present
        # loc = '/home/daankrol/Dataset-Reduction-IL/knn.pickle'
        # if os.path.isfile(loc):
        #     with open(loc, 'rb') as f:
        #         print('Loading pickle file.')
        #         self.knn = pickle.load(f)
        #     return
        # else:
        #     print('cannot find knn pickel file.')
        self.logger.info('Finding k-nearest-neighbours')
        self.get_labels()

        # for each sample in a class, compute distance in feature space to all other sample in the same class
        # and find the k-nearest-neighbours

        # knn like: knn[class][sample] = [k-nearest-neighbours], which can be variable length vector of indices in the original dataset

        knn = []
        for c in range(self.num_classes):
            # add new empty list to knn for the current class 
            knn.append([])
            class_indices = np.where(self.trn_lbls == c)[0]
            embeddings = torch.zeros((len(class_indices), self.pretrained_model.embDim)).to(self.device)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(self.trainloader.dataset, class_indices),
                batch_size=self.trainloader.batch_size,
                pin_memory=self.trainloader.pin_memory, num_workers=self.trainloader.num_workers)
            for i, (inputs, _) in enumerate(loader):
                inputs = inputs.to(self.device)
                _, e = self.pretrained_model(inputs, last=True, freeze=True)
                embeddings[i * self.trainloader.batch_size:(i + 1) * self.trainloader.batch_size] = e.detach()

            # calculate pairwise distances and for each sample, find the k-nearest-neighbours, add their indices sorted by their distance
            dist = pairwise_distances(embeddings.cpu().numpy())

            for i in range(len(class_indices)):
                # can happen that we have less than k samples in a class, then we just take all samples and 
                neighbours = np.argsort(dist[i])[1:self.k + 1].astype(np.int32)
                knn[c] += list(neighbours)
                    # np.append(knn[c], neighbours)

        del self.pretrained_model  # only need this at the start.
        self.knn = knn


    @torch.no_grad()
    def calculate_divergence(self):
        # We use the current training model to determine the probabilities
        self.logger.info('Calculating probabilities for divergence')
        self.model.eval()
        probs = []
        # probs[c][class_indexed sample] = [prob vector] 
        # knn[c][class_indexed sample] = [class_indexed k-nearest-neighbours]
        cnt = 0
        for c in range(self.num_classes):
            probs.append([])
            class_indices = np.where(self.trn_lbls == c)[0]
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(self.trainloader.dataset, class_indices),
                batch_size=self.trainloader.batch_size,
                pin_memory=self.trainloader.pin_memory, num_workers=self.trainloader.num_workers)
            for i, (inputs, _) in enumerate(loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs, freeze=True)
                outputs = F.softmax(outputs, dim=1).detach().cpu().numpy()

                # calculate probs for this batch. Add all vectors in dim 0 to probs without flattening
                # have to use python lists as they have variable length.
                probs[c] += list(outputs)

        self.logger.debug('Calculated probabilities, now computing divergence')
        # calculate divergence
        divergence = np.zeros(len(self.trn_lbls))
        for c in range(self.num_classes):
            # Calculate KL-divergence between sample PDF and all neighbours of the same class.
            # Average the divergence scores over all neighbours.
            class_indices = np.where(self.trn_lbls == c)[0]
            for i in range(len(class_indices)):
                aa = np.array(probs[c][i])

                try:
                    # Some samples have no neighbours since the classes can be heavily imbalanced.
                    # Use a very low (0) divergence score for those.
                    neighbours = self.knn[c][i]
                    neighbour_probs = np.array(probs[c][neighbours])
                except Exception as e:
                    # print(c, i)
                    # print(len(self.knn[c]), self.knn[c])
                    neighbour_probs = aa
                divergence[class_indices[i]] = np.mean(np.sum(neighbour_probs * np.log(neighbour_probs / aa), axis= 1 if neighbour_probs.ndim>1 else 0 ))
                # if divergence[class_indices[i]] == 0:
                #     l = np.log(neighbour_probs / aa)
                #     nl = neighbour_probs * l
                #     su = np.sum(nl, axis=1 if neighbour_probs.ndim > 1 else 0)
                #     print('\n', c,i,  l, nl, su, aa, neighbour_probs)
        self.model.train()
        self.logger.debug('Divergence calculated.')
        return divergence

    def select(self, budget, model_params):
        start_time = time.time()
        self.logger.info(f'Started {self.selection_type} Super-CL selection.')
        self.update_model(model_params)
        self.find_knn()
        self.fraction = budget / self.N_trn
        divergence = self.calculate_divergence()

        # Higher score means closer to decision boundary and thus more important
        if self.selection_type == 'PerClass':
            indices = np.array([], dtype=np.int32)
            for c in range(self.num_classes):
                class_indices = np.where(self.trn_lbls == c)[0]
                num_samples = int(self.fraction * len(class_indices))
                # add the indices of the selected samples that have the highest KL divergence
                indices = np.append(indices,
                                    class_indices[np.argsort(divergence[class_indices])[-num_samples:]])
        else:
            # add the indices of the selected samples that have the highest KL divergence
            indices = np.argsort(divergence)[-int(self.fraction * self.N_trn):]

        end_time = time.time()
        self.logger.info(f'Super-CL algorithm took {end_time - start_time} seconds.')
        self.logger.info('Selected {}  samples with a budget of {}'.format(len(indices), budget))
        print(indices)
        print(divergence[indices])
        print(np.count_nonzero(divergence[indices] == 0), '/', len(divergence[indices]))
        # if self.weighted:
        #     return indices, divergence[indices].astype(np.float32)

        return indices, torch.ones(len(indices))
