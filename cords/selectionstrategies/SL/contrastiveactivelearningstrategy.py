import apricot
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import pairwise_distances
import time
from cords.utils.models.efficientnet import EfficientNetB0_PyTorch

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
                 device, num_classes, selection_type, logger, metric='euclidean', k=10):
        super().__init__(trainloader, valloader, model, num_classes,None, loss, device, logger)
        self.selection_type = selection_type
        self.k = k
        self.pretrained_model = EfficientNetB0_PyTorch(num_classes=self.num_classes, pretrained=True, fine_tune=False).to(self.device)
        self.pretrained_model.eval()
        if metric == 'euclidean':
            self.metric = self.euclidean_distance_scipy
        elif metric == 'cossim':
            self.metric = lambda a, b: -1. * self.cossim_pair_np(a, b)
        else:
            raise ValueError('Invalid metric')

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


    def euclidean_dist_pair_torch(self, x):
        # if x is a numpy array, convert it to torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        (rowx, colx) = x.shape
        xy = torch.mm(x, x.t())
        x2 = torch.sum(torch.mul(x, x), dim=1).reshape(-1, 1)
        return torch.sqrt(torch.clamp(x2 + x2.t() - 2. * xy, min=1e-12))

    @torch.no_grad()
    def find_knn(self):
        "Find k-nearest-neighbour datapoints based on feature embedding by a pretrained network"
        # NOTE: this is done since in normal Active Learning we don't have the label information. 
        # TODO: Could use distance to mean feature embedding since we have label info. 
        self.logger.info('Finding k-nearest-neighbours')
        self.logger.info('Computing feature embedding')

        if self.selection_type == 'PerClass':
            self.get_labels()
            knn = []
            for c in range(self.num_classes):
                class_indices = np.arange(self.N_trn)[self.trn_lbls == c]
                embeddings = torch.zeros((len(class_indices), self.pretrained_model.embDim)).to(self.device)
                loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.trainloader.dataset, class_indices),
                                                        batch_size=self.trainloader.batch_size)
                for i, (inputs, _) in enumerate(loader):
                    inputs = inputs.to(self.device)
                    _, embedding = self.pretrained_model(inputs, last=True, freeze=True)
                    embedding = embedding.detach().cpu().numpy()
                    # Embedding is of shape (batch_size, embDim)                    
                    embeddings[i * self.trainloader.batch_size: i * (self.trainloader.batch_size + inputs.shape[0])] = embedding
                    # the row above will fail since the embedding is of size (batch_size, embDim)
                    # error = the expanded size of the tensor (0) must match the existing size (32) at non-singleton dimension 0
                    # target size = (0, 1280)
                    # actual tensor size = (32, 1280)

                    # embeddings[i*self.trainloader.batch_size: i*(self.trainloader.batch_size+inputs.shape[0])] = 


                dist = self.metric(embeddings.cpu().numpy())
                knn.append(np.argsort(dist, axis=1)[:, 1:self.k + 1])
            self.logger.info('Finished with computing embeddings')
            return knn
        else:
            embeddings = torch.zeros((self.N_trn, self.pretrained_model.embDim)).to(self.device)
            loader = torch.utils.data.DataLoader(self.trainloader.dataset, batch_size=self.trainloader.batch_size)
            for i, (inputs, _) in enumerate(loader):
                inputs = inputs.to(self.device)
                _, embedding = self.pretrained_model(inputs, last=True, freeze=True)
                embedding = embedding.detach()
                embeddings[i*self.trainloader.batch_size:(i*self.trainloader.batch_size + inputs.shape[0]) ] = embedding
                if i % 20 == 0:
                    self.logger.debug('Computed embedding for {}/{}'.format(i, len(loader)))
            self.logger.info('Finished with computing embeddings')
            dist = self.metric(embeddings.cpu().numpy())
            return np.argsort(dist, axis=1)[:, 1:self.k+1]

    @torch.no_grad()
    def calculate_KL_divergence(self, knn, index=None):
        # We use the current training model to determine the probabilities of the k-nearest-neighbours
        self.logger.info('Calculating KL divergence')
        self.model.eval()

        if index is None:
            loader = torch.utils.data.DataLoader(self.trainloader.dataset, batch_size=self.trainloader.batch_size)
        else:
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.trainloader.dataset, index),
                                                    batch_size=self.trainloader.batch_size)
        probs = torch.zeros([len(loader.dataset), self.num_classes]).to(self.device)
        batch_num = len(loader)
        batch_size = loader.batch_size
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(self.device)
            # last batch can have different size
            cur_batch_size = labels.shape[0]
            # save last batch size 
            if i == batch_num - 1:
                last_batch_size = cur_batch_size
            probs[i* batch_size: i*batch_size + cur_batch_size] = torch.nn.functional.softmax(self.model(inputs, freeze=True), dim=1).detach()

        kl = torch.zeros(batch_num).to(self.device)
        print(f'starting with KL divergence. Total num batches {batch_num} with batch size {batch_size}')
        for i in range(0, batch_num, batch_size):
            # the last batch might be smaller than batch_size
            batch_size = batch_size if i < batch_num - batch_size else last_batch_size
            print(f'kl-for batch {i}')
            aa = probs[i*batch_size: i*batch_size + batch_size].unsqueeze(1)
            bb = probs[knn[i*batch_size: i*batch_size + batch_size]].unsqueeze(0)
            kl[i:(i+batch_size)] = torch.sum(0.5 * aa * torch.log(aa / bb) + 0.5 * bb * torch.log(bb/aa), dim=2).mean(dim=1)
        
            # aa = np.expand_dims(probs[i:(i+batch_size)], 1).repeat(self.k, 1)
            # bb = probs[knn[i:(i+batch_size)], :]
            # kl[i:(i+batch_size)] = np.mean(np.sum( 0.5 * aa * np.log(aa/bb) + 0.5 * bb * np.log(bb/aa), axis=2), axis=1)
        
        self.model.train()
        return kl.cpu().numpy()


    def select(self, budget, model_params):
        start_time = time.time()
        self.logger.info(f'Started {self.selection_type} CAL selection.')
        self.update_model(model_params)
        self.knn = self.find_knn()
        self.fraction = budget / self.N_trn
        indices = np.array([], dtype=np.int32)

        # Higher score means closer to decision boundary and thus more important
        if self.selection_type == 'PerClass':
            for c in range(self.num_classes):
                class_indices = torch.arange(self.N_trn)[self.trn_lbls == c]
                scores = self.calculate_KL_divergence(self.knn, index=class_indices)
                # add the indices of the selected samples that have the highest KL divergence
                indices = np.concatenate((indices, class_indices[np.argsort(scores)[-int(self.fraction * len(class_indices)):]]))
        else:
            scores = self.calculate_KL_divergence(self.knn)
            indices = np.argsort(scores)[-int(self.fraction * self.N_trn):]

        end_time = time.time()
        self.logger.info(f'CAL algorithm took {end_time - start_time} seconds.')
        self.logger.info('Selected {}  samples with a budget of {}'.format(len(indices), budget))
        return indices, torch.ones(len(indices))
