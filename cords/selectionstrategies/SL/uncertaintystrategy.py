from .dataselectionstrategy import DataSelectionStrategy
import time
import torch
import torch.nn.functional as F
import numpy as np


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

    def select(self, budget, model_params):
        start_time = time.time()
        self.model.load_state_dict(model_params)
        self.logger.info(f"Started {self.selection_type} uncertainty selection")
        self.logger.info("Budget: {0:d}".format(budget))

        if self.selection_type == "LeastConfidence":
            idx, gamma = self.leastConfidenceSelection(budget)
        elif self.selection_type == "MarginOfConfidence":
            idx, gamma = self.marginOfConfidenceSelection(budget)
        elif self.selection_type == "Entropy":
            idx, gamma = self.entropySelection(budget)

        end_time = time.time()
        self.logger.info(
            "Uncertainty algorithm Subset Selection time is: {0:.4f}".format(
                end_time - start_time
            )
        )
        return idx, gamma

    def leastConfidenceSelection(self, budget):
        probs = torch.zeros([self.N_trn, self.num_classes]).to(self.device)
        indices = torch.arange(self.N_trn).to(self.device)
        evaluated_samples = 0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.trainloader):
                inputs = inputs.to(self.device)
                out = self.model(inputs, freeze=True)
                out = F.softmax(out, dim=1)

                #   least confidence: difference between the most confident and 100% confidence
                start_slice = evaluated_samples
                end_slice = evaluated_samples + out.shape[0]
                probs[start_slice:end_slice] = out
                evaluated_samples = end_slice

        probs = probs.max(1)[0]
        # sort the indices by ascending confidence using torch.sort
        indices = indices[torch.argsort(probs, dim=0)]
        print('indices: ', indices[:10])

        return indices[:budget].cpu().numpy(), torch.ones(budget)


    def marginOfConfidenceSelection(self, budget):
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

    def entropySelection(self, budget):
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
