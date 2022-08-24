import apricot
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data.sampler import SubsetRandomSampler


class ContrastiveActiveLearningStrategy(DataSelectionStrategy):
    """
    This class extends :class:`selectionstrategies.supervisedlearning.dataselectionstrategy.DataSelectionStrategy`
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
    linear_layer: bool
        Apply linear transformation to the data
    if_convex: bool
        If convex or not
    selection_type: str
        PerClass or PerBatch
    """

    def __init__(self, trainloader, valloader, model, loss,
                 device, num_classes, linear_layer, selection_type, logger, metric='euclidean', k=10):
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device, logger)
        self.if_convex = if_convex
        self.selection_type = selection_type
        self.k = k
        if metric == 'euclidean':
            self.metric = self.euclidean_dist_pair_np
        elif metric == 'cossim':
            self.metric = lambda a, b: -1. * cossim_pair_np(a, b)
        else:
            raise ValueError('Invalid metric')

    def cossim_pair_np(self, v1):
        num = np.dot(v1, v1.T)
        norm = np.linalg.norm(v1, axis=1)
        denom = norm.reshape(-1, 1) * norm
        res = num / denom
        res[np.isneginf(res)] = 0.
        return 0.5 + 0.5 * res

    def euclidean_dist_pair_np(self, x):
        (rowx, colx) = x.shape
        xy = np.dot(x, x.T)
        x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowx, axis=1)
        return np.sqrt(np.clip(x2 + x2.T - 2. * xy, 1e-12, None))

    def find_knn(self, model_params):
        "Find k-nearest-neighbour datapoints based on feature embedding"
        self.logger.info('Finding k-nearest-neighbours')
        self.logger.info('Computing feature embedding')
        self.model.load_state_dict(model_params)
        self.model.eval()

        if self.selection_type == 'PerClass':
            self.get_labels()
            knn = []
            for c in range(self.num_classes):
                class_indices = np.arange(self.N_trn)[self.trn_lbls == c]
                embeddings = []
                loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.trainloader.dataset, class_indices),
                                                        batch_size=self.trainloader.batch_size)
                for i, (inputs, _) in enumerate(loader):
                    inputs = inputs.to(self.device)
                    embeddings.append(self.model(inputs, last=True, freeze=True).cpu().numpy())
                
                embeddings = np.concatenate(embeddings, axis=0)
                knn.append(np.argsort(self.metric(embeddings), axis=1)[:, 1:self.k+1])
            return knn
        else:
            embeddings = []
            trainset = self.trainloader.sampler.data_source
            loader = self.trainloader
            loader_num = len(loader)
            for i, (inputs, _) in enumerate(loader):
                inputs = inputs.to(self.device)
                out, embedding = self.model(inputs, last=True, freeze=True)
                embeddings.append(embedding)
                if i % 100 == 0:
                    self.logger.info('Computed embedding for {}/{}'.format(i, loader_num))

            # embeddings = np.concatenate(arrays=embeddings, axis=0)
            # return np.argsort(self.metric(embeddings), axis=1)[:, 1:(self.k+1)]
            embeddings = torch.cat(embeddings, dim=0)
            return np.argsort(self.metric(embeddings.cpu().numpy()), axis=1)[:, 1:self.k + 1]

    def calculate_KL_divergence(self, knn, index=None):
        self.model.eval()
        with torch.no_grad():
            if index is None:
                loader = self.trainloader
            else:
                loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.trainloader.dataset, index),
                                                        batch_size=self.trainloader.batch_size)

            probs = np.zeros([loader.dataset.__len__(), self.num_classes])
            batch_num = len(loader)
            batch_size = loader.batch_size
            for i, (inputs, labels) in enumerate(loader):
                probs[i* batch_size: (i+1)*batch_size] = torch.nn.functional.softmax(self.model(inputs, freeze=True)[0], dim=1).detach().cpu().numpy()

            s = np.zeros(batch_num)
            for i in range(0, batch_num, batch_size):
                aa = np.expand_dims(probs[i:(i+batch_size)], 1).repeat(self.k, 1)
                bb = probs[knn[i:(i+batch_size)], :]
                s[i:(i+batch_size)] = np.mean(np.sum( 0.5 * aa * np.log(aa/bb) + 0.5 * bb * np.log(bb/aa), axis=2), axis=1)
            return s


    def select(self, budget, model_params):
        start_time = time.time()
        self.knn = self.find_knn(model_params)
        self.logger.info(f'Started {self.selection_type} CAL selection.')
        self.logger.info('Calculating KL divergence')
        self.fraction = budget / self.N_trn
        scores = []
        indices = np.array([], dtype=np.int32)

        if self.selection_type == 'PerClass':
            for c in range(self.num_classes):
                class_indices = np.arange(self.N_trn)[self.trn_lbls == c]
                scores.append(self.calculate_KL_divergence(self.knn, index=class_indices))
                indices = np.append(indices, class_indices[np.argsort(scores[-1])[::1][:round(len(class_index) * self.fraction)]])
        else:
            indices = np.argsort(self.calculate_KL_divergence(self.knn))[::1][:budget]

        end_time = time.time()
        self.logger.info(f'CAL algorithm took {end_time - start_time} seconds.')
        self.logger.info('Selected {}  samples with a budget of {}'.format(len(indices), budget))
        return indices, torch.ones(len(indices))
