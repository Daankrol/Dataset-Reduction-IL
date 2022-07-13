import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pandas as pd
import numpy as np
import wandb
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


"""
This class is used to build image embedding that can be used for TSNE plots.
"""


class TSNEPlotter:
    def __init__(
        self, full_trainloader, full_valloader, full_testloader, subset_indices, device
    ):
        # use pretrained imagenet efficientnet b0 model as feature extractor
        self.model = torchvision.models.efficientnet_b0(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.full_trainloader = full_trainloader
        self.full_valloader = full_valloader
        self.full_testloader = full_testloader
        self.subset_indices = subset_indices
        self.embDim = 1280
        self.train_embeddings = None
        self.val_embeddings = None
        self.test_embeddings = None
        self.df = None
        self.tsne = TSNE(
            n_components=2, perplexity=30, init="pca", learning_rate="auto", n_jobs=-1
        )

        # remove the classification head from the model as we only use it for embedding
        self.model.classifier = nn.Sequential()
        self.construct_embeddings()

    def construct_embeddings(self):
        # Create embeddings for the whole training set
        self.train_embeddings = self._make_embedding_for_dataloader(
            self.full_trainloader
        )
        cols = [f"out_{i}" for i in range(self.train_embeddings.shape[1])]
        self.train_labels = [y for x, y in self.full_trainloader.dataset]

        # create tSNE plot with pca embeddings
        time_start = time.time()

        self.tsne_embeddings = self.tsne.fit_transform(self.train_embeddings)
        self.df = pd.DataFrame(self.tsne_embeddings, columns=["tsne-2d-x", "tsne-2d-y"])
        self.df["LABEL"] = self.train_labels
        self.df["tsne-2d-x"] = self.tsne_embeddings[:, 0]
        self.df["tsne-2d-y"] = self.tsne_embeddings[:, 1]

        print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))

    def make_tsne_plot(self, epoch, selected_indices=None):
        # reset all the selected indices to False
        # if a row is selected, mark it as selected in the dataframe
        if selected_indices is not None:
            self.df["selected_indices"] = False
            self.df.loc[selected_indices, "selected_indices"] = True

        table = wandb.Table(columns=self.df.columns.tolist(), data=self.df.values)
        wandb.log({"tSNE_data": table}, step=epoch)

        # plot the tSNE plot
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-x",
            y="tsne-2d-y",
            hue="LABEL",
            palette=sns.color_palette("hls", len(self.df["LABEL"].unique())),
            data=self.df,
            legend="full",
            alpha=0.7,
        )

        # visualize every selected point
        if selected_indices is not None:
            plt.scatter(
                self.df["tsne-2d-x"][selected_indices],
                self.df["tsne-2d-y"][selected_indices],
                marker="*",
                color="black",
                s=80,
            )
        plt.title("t-SNE plot")

        # random file name to save the plot
        file_name = f"tSNE_plot_{time.time()}.png"
        plt.savefig(file_name)
        wandb.log({"tSNE_plot": wandb.Image(file_name)}, step=epoch)
        plt.close()
        # if file is already there, delete it
        if os.path.exists(file_name):
            os.remove(file_name)

    def _make_embedding_for_dataloader(self, dataloader):
        # Create embeddings for the given batched dataloader
        with torch.no_grad():
            embeddings = torch.zeros((len(dataloader.dataset), self.embDim))
            # note that the dataloader is a torch.utils.data.DataLoader object
            # and that the data is batched
            for i, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                embeddings[
                    i * dataloader.batch_size : (i + 1) * dataloader.batch_size
                ] = self._get_embedding(images)

        return embeddings

    def _get_embedding(self, images):
        # make sure that the image is in the right size
        # when reshaping make sure that the batch remains
        if images.size()[2] != 224 or images.size()[3] != 224:
            images = torch.stack(
                [
                    nn.functional.interpolate(
                        image.unsqueeze(0),
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )
                    for image in images
                ],
                dim=0,
            )
            # This stack adds a new dimension to the tensor, we should remove it
            images = images.squeeze(1)
        # get the embedding for the images
        with torch.no_grad():
            out = self.model.features(images)
            out = F.adaptive_avg_pool2d(out, 1)
            e = out.view(out.size(0), -1)
        return e


if __name__ == "__main__":
    # make two random images with resolutaion 32x32
    images = torch.randn(50, 3, 32, 32)
    # make a tensor with 50 random labels between 0 and 9
    labels = torch.randint(0, 10, (50,))

    # make a tensor dataset and a dataloader
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    tp = TSNEPlotter(dataloader, dataloader, dataloader, None, "cpu")
    tp.make_tsne_plot(0, [1, 2, 3, 4, 5])
    tp.make_tsne_plot(1, [6, 7, 8, 9, 10])
    tp.make_tsne_plot(2, [1, 2, 3, 7, 8, 9])
