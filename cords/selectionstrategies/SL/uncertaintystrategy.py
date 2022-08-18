from .dataselectionstrategy import DataSelectionStrategy
import time
import torch
import torch.nn.functional as F
import numpy as np


# Adapted from: https://github.com/PatrickZH/DeepCore/blob/main/deepcore/methods/uncertainty.py 

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
        selection_type,
        logger,
        balance=False
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
        self.selection_type = selection_type
        self.balance = balance

        if self.selection_type == "LeastConfidence":
            self.method = self.leastConfidenceSelection
        elif self.selection_type == "MarginOfConfidence":
            self.method = self.marginOfConfidenceSelection
        elif self.selection_type == "Entropy":
            self.method = self.entropySelection
        else:
            raise NotImplementedError("Selection algorithm not implemented")

    def select(self, budget, model_params):
        start_time = time.time()
        self.model.load_state_dict(model_params)
        self.logger.info(f"Started {self.selection_type} uncertainty selection." + " With per class balancing" if self.balance else "")
        self.logger.info("Budget: {0:d}".format(budget))
        self.fraction = budget / self.N_trn
        if self.balance:
            self.logger.info('Uncertainty balancing with fraction: {0:.2f}'.format(self.fraction))
        
        if self.balance:
            # per-class sampling
            self.get_labels()
            indices = np.array([], dtype=np.int64)
            scores = []
            # we want to fill the budget with samples from each class
            for c in range(self.num_classes):
                class_index = np.arange(self.N_trn)[self.trn_lbls == c]
                scores.append(self.rank_uncertainty(class_index))
                indices = np.append(indices, class_index[np.argsort(scores[-1])[
                                                               :round(len(class_index) * self.fraction)]])
        else:
            scores = self.method(budget)
            indices = np.argsort(scores)[::-1][:budget]


        end_time = time.time()
        self.logger.info(
            "Uncertainty algorithm Subset Selection time is: {0:.4f}.".format(
                end_time - start_time
            )
        )
        self.logger.info("Selected {} samples with a budget of {}".format(len(indices), budget))

        return indices, torch.ones(len(indices))

    def rank_uncertainty(self, index=None):
        self.model.eval()
        with torch.no_grad():
            if index is None:
                loader = self.trainloader
            else:
                loader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(self.trainloader.dataset, index),
                    batch_size=self.trainloader.batch_size)
            
            scores = np.array([])
            batch_num = len(loader)

            for i, (input, _) in enumerate(loader):
                if self.selection_type == "LeastConfidence":
                    scores = np.append(scores, self.model(input.to(self.device), freeze=True).max(axis=1).values.cpu().numpy())
                elif self.selection_type == "MarginOfConfidence":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.device)), dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[
                        torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
                elif self.selection_type == "Entropy":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.device)), dim=1).cpu()
                    scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
        return scores



    def leastConfidenceSelection(self, index=None):
        self.model.eval()
        with torch.no_grad():
            scores = np.array([])
            batch_num = len(self.trainloader)
            for i, (inputs, _) in enumerate(self.trainloader):
                scores = np.append(scores, self.model(inputs.to(self.device), freeze=True).max(axis=1).values.cpu().numpy())
        return scores
        # indices = np.argsort(scores)[::-1][:budget]
        # return indices, torch.ones(len(indices))

    def marginOfConfidenceSelection(self, index=None):
        # Margin of confidence is defined by the difference between the top two confidence values
        # source: https://github.com/PatrickZH/DeepCore/blob/main/deepcore/methods/uncertainty.py
        self.model.eval()
        with torch.no_grad():
            scores = np.array([])
            batch_num = len(self.trainloader)
            for i, (input, _) in enumerate(self.trainloader):
                preds = torch.nn.functional.softmax(self.model(input.to(self.device)), dim=1)
                preds_argmax = torch.argmax(preds, dim=1)
                max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                preds_sub_argmax = torch.argmax(preds, dim=1)
                scores = np.append(scores, (max_preds - preds[
                    torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
        return scores

    def entropySelection(self, index=None):
        self.model.eval()
        with torch.no_grad():
            scores = np.array([])
            batch_num = len(self.trainloader)
            for i, (input, _) in enumerate(self.trainloader):
                preds = torch.nn.functional.softmax(self.model(input.to(self.device)), dim=1).cpu().numpy()
                scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
        return scores


    def _marginOfConfidenceSelectionOld(self, budget):
        # Margin of confidence is defined by the difference between the top two confidence values
        idxs = []
        margins = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.trainloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                out = self.model(inputs, freeze=True)
                out = F.softmax(out, dim=1)
                # margin: difference between the top 2 most confident
                top2 = torch.topk(out, 2, dim=1)[0]
                margin = top2[:, 1] - top2[:, 0]
                margins.extend(margin.cpu().numpy())
                idxs.extend(i * np.arange(0, margin.shape[0]))

        # sort the idxs by ascending margin
        idxs = np.array(idxs)
        margins = np.array(margins)
        idxs = idxs[np.argsort(margins)]
        margins = margins[np.argsort(margins)]

        idxs = idxs[:budget]
        gammas = torch.ones(len(idxs))
        return idxs, gammas

    def _entropySelectionOld(self, budget):
        idxs = []
        entropies = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.trainloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                out = self.model(inputs, freeze=True)
                out = F.softmax(out, dim=1)
                # entropy:
                entropy = -torch.sum(out * torch.log(out), dim=1)
                entropies.extend(entropy.cpu().numpy())
                idxs.extend(i * np.arange(0, entropy.shape[0]))

        # sort the idxs by ascending entropy
        idxs = np.array(idxs)
        entropies = np.array(entropies)
        idxs = idxs[np.argsort(entropies)]
        entropies = entropies[np.argsort(entropies)]

        idxs = idxs[:budget]
        gammas = torch.ones(len(idxs))
        return idxs, gammas