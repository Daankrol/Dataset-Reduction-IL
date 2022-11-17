import math
import numpy as np
import torch
import torch.nn.functional as F
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from sklearn.metrics import pairwise_distances
import time
from cords.utils.models.efficientnet import EfficientNetB0_PyTorch
import faiss
from scipy.spatial.distance import jensenshannon


class ContrastiveActiveLearningStrategy(DataSelectionStrategy):
    """
    This class extends :class:`selectionstrategises.supervisedlearning.dataselectionstrategy.DataSelectionStrategy`
    to use Contrastive Active Learning (CAL) as a data selection strategy.

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
                 device, num_classes, selection_type, logger, k=10, weighted=False, use_faiss=True):
        super().__init__(trainloader, valloader, model, num_classes, None, loss, device, logger)
        self.selection_type = selection_type
        self.weighted = weighted
        self.k = k
        self.knn = None
        self.use_faiss = use_faiss
        self.pretrained_model = EfficientNetB0_PyTorch(num_classes=self.num_classes, pretrained=True,
                                                       fine_tune=False).to(self.device)
        # disable gradients for the pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.eval()
        self.metric = self.euclidean_distance_scipy
        if self.use_faiss:
            self.faiss_index = faiss.IndexFlatL2(self.pretrained_model.embDim)

    def cossim_pair_np(self, v1):
        num = np.dot(v1, v1.T)
        norm = np.linalg.norm(v1, axis=1)
        denom = norm.reshape(-1, 1) * norm
        res = num / denom
        res[np.isneginf(res)] = 0.
        return 0.5 + 0.5 * res

    def euclidean_distance_scipy(self, x):
        return pairwise_distances(x, metric='euclidean', n_jobs=-1)

    def euclidean_dist_pair_torch(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        (rowx, colx) = x.shape
        xy = torch.mm(x, x.t())
        x2 = torch.sum(torch.mul(x, x), dim=1).reshape(-1, 1)
        return torch.sqrt(torch.clamp(x2 + x2.t() - 2. * xy, min=1e-12))

    def euclidean_dist_pair_np(self, x):
        (rowx, colx) = x.shape
        xy = np.dot(x, x.T)
        x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowx, axis=1)
        return np.sqrt(np.clip(x2 + x2.T - 2. * xy, 1e-12, None))

    def kl_divergence(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    @torch.no_grad()
    def find_knn(self):
        "Find k-nearest-neighbour datapoints based on feature embedding by a pretrained network"
        # NOTE: this is done since in normal Active Learning we don't have the label information.
        if self.knn is not None:
            return
        self.logger.info('Finding k-nearest-neighbours')
        self.logger.debug('Computing feature embedding')

        if self.use_faiss:
            # construct feature space, make sure the loader does not shuffle as this will change the order of
            # Faiss vector IDs
            loader = torch.utils.data.dataloader.DataLoader(self.trainloader.dataset,batch_size=self.trainloader.batch_size,
                                                         pin_memory=self.trainloader.pin_memory,
                                                         num_workers=self.trainloader.num_workers,
                                                            shuffle=False)
            embeddings = []
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(self.device)
                _, embedding = self.pretrained_model(inputs, last=True, freeze=True)
                embedding = embedding.detach()
                embeddings.append(embedding.flatten(1).cpu().numpy())
            embeddings = np.concatenate(embeddings, axis=0)
            self.logger.debug("Constructed feature space. Now building Faiss index")
            self.faiss_index.add(embeddings)
            self.logger.debug(f"Index built. Starting knn search with Faiss. K={self.k}")
            D, I = self.faiss_index.search(embeddings, self.k+1)
            self.knn = I[:, 1:self.k+1]
        else:
            if self.selection_type == 'PerClass':
                self.get_labels()
                knn = []
                for c in range(self.num_classes):
                    class_indices = np.arange(self.N_trn)[self.trn_lbls == c]
                    # embeddings = torch.zeros((len(class_indices), self.pretrained_model.embDim)).to(self.device)
                    embeddings = []
                    loader = torch.utils.data.DataLoader(Subset(self.trainloader.dataset, class_indices),
                                                         batch_size=self.trainloader.batch_size,
                                                         pin_memory=self.trainloader.pin_memory,
                                                         num_workers=self.trainloader.num_workers)
                    for i, (inputs, _) in enumerate(loader):
                        inputs = inputs.to(self.device)
                        _, embedding = self.pretrained_model(inputs, last=True, freeze=True)
                        embedding = embedding.detach()
                        # Embedding is of shape (batch_size, embDim)
                        # embeddings[
                        # i * self.trainloader.batch_size: (i * self.trainloader.batch_size + inputs.shape[0])] = embedding
                        embeddings.append(embedding.flatten(1).cpu().numpy())
                    embeddings = np.concatenate(embeddings, axis=0)

                    # calculate pairwise distance matrix
                    dist = self.metric(embeddings)
                    # for each sample add the k nearest neighbours
                    knn.append(np.argsort(dist, axis=1)[:, 1:self.k + 1])
                self.logger.debug('Finished with computing embeddings and distances')
            else:
                # embeddings = torch.zeros((self.N_trn, self.pretrained_model.embDim)).to(self.device)
                embeddings = []
                # self.loader = torch.utils.data.DataLoader(self.trainloader.dataset, batch_size=self.trainloader.batch_size,
                #                                      pin_memory=self.trainloader.pin_memory,
                #                                      num_workers=self.trainloader.num_workers)
                for i, (inputs, _) in enumerate(self.trainloader):
                    inputs = inputs.to(self.device)
                    _, embedding = self.pretrained_model(inputs, last=True, freeze=True)
                    embeddings.append(embedding.detach().flatten(1).cpu().numpy())
                embeddings = np.concatenate(embeddings, axis=0)      # flatten all batches
                self.logger.info('Finished computing embeddings, starting distance calculation')
                knn = np.argsort(self.metric(embeddings), axis=1)[:, 1:self.k + 1]
                self.logger.debug('Finished with computing embeddings and distances')
            del embeddings
            self.knn = knn

    @torch.no_grad()
    def calculate_divergence(self, index=None):
        # We use the current training model to determine the probabilities of the k-nearest-neighbours
        self.logger.info('Calculating divergence')
        self.model.eval()

        loader = torch.utils.data.dataloader.DataLoader(self.trainloader.dataset,
                                                        batch_size=self.trainloader.batch_size,
                                                        pin_memory=self.trainloader.pin_memory,
                                                        num_workers=self.trainloader.num_workers,
                                                        shuffle=False)
        probs = np.zeros([self.N_trn, self.num_classes])
        bsize = loader.batch_size
        for i, (inputs, targets) in enumerate(loader):
            probs[i * bsize: (i+1) * bsize] = torch.nn.functional.softmax(
                self.model(inputs.to(self.device)), dim=1).detach().cpu().numpy()
        s = np.zeros(self.N_trn)

        for i in range(0, self.N_trn, bsize):
            aa = np.expand_dims(probs[i:(i+bsize)], 1).repeat(self.k, 1)
            # aa is prob vector of batch. Where each sample pdf is repeated k times.
            n = self.knn[i:(i+bsize)]
            bb = probs[n, :]
            jsd = jensenshannon(aa, bb, axis=-1) ** 2  # power 2, to go from distance to divergence.
            s[i:(i+bsize)] = np.mean(jsd, axis=-1)

        self.model.train()
        return s

    def select(self, budget, model_params):
        start_time = time.time()
        self.logger.info(f'Started {"FAISS per-batch" if self.use_faiss else  self.selection_type} CAL selection.')
        self.update_model(model_params)
        self.find_knn()
        self.fraction = budget / self.N_trn
        indices = np.array([], dtype=np.int32)

        # Higher score means closer to decision boundary and thus more important
        if self.selection_type == 'PerClass':
            for c in range(self.num_classes):
                class_indices = torch.arange(self.N_trn)[self.trn_lbls == c]
                scores = self.calculate_divergence(index=class_indices)
                # add the indices of the selected samples that have the highest KL divergence
                indices = np.concatenate(
                    (indices, class_indices[np.argsort(scores)[-int(self.fraction * len(class_indices)):]]))
        else:
            scores = self.calculate_divergence()
            indices = np.argsort(scores)[-math.ceil(self.fraction * self.N_trn):]

        end_time = time.time()
        self.logger.info(f'CAL algorithm took {end_time - start_time} seconds.')
        self.logger.info('Selected {}  samples with a budget of {}'.format(len(indices), budget))

        if self.weighted:
            weights = scores[indices]
            return indices, weights

        return indices, torch.ones(len(indices))
