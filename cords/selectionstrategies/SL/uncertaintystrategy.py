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

    def select(self, budget, model_params):
        start_time = time.time()
        self.model.load_state_dict(model_params)
        self.logger.info("Started uncertainty selection")
        self.logger.info("Budget: {0:d}".format(budget))
        idxs = []
        uncertainties = []
        gammas = []
        # Compute the uncertainty of the model on all training data
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                # turn this into probability
                out = F.softmax(out, dim=1)
                # difference between maximal confidence (1) and confidence of the actual label
                # gamma is for the max class but we want it for the actual class so we need to take the index of the max
                # and then use the index of the actual class
                uncertainty = 1 - out[:, y].item()
                idxs.append(i)
                uncertainties.append(uncertainty)

        # sort the idxs by descending uncertainty
        idxs = np.array(idxs)
        uncertainties = np.array(uncertainties)
        idxs = idxs[uncertainties.argsort()[::-1]]
        uncertainties = uncertainties[uncertainties.argsort()[::-1]]

        # select the top k samples where k == budget
        # return the indices of the selected samples
        idxs = idxs[:budget]
        gammas = torch.ones(idxs)
        end_time = time.time()
        self.logger.info(
            "Uncertainty algorithm Subset Selection time is: {0:.4f}".format(
                end_time - start_time
            )
        )
        return selected_idxs, gammas
