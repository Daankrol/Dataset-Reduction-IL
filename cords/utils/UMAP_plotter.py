import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pandas as pd
import numpy as np
import wandb
import time
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import umap
from sklearn.manifold import TSNE

"""
This class is used to build image embedding that can be used for UMAP plots.
"""


class UMAPPlotter:
    def __init__(
        self, full_trainloader, full_valloader, full_testloader, subset_indices, device, root, dataset_name
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
        self.umap = umap.UMAP(n_neighbors=10, n_components=2)

        # remove the classification head from the model as we only use it for embedding
        self.model.classifier = nn.Sequential()

        self.root = os.path.join(root, 'embeddings')
        self.ds_name = dataset_name

        # check if embeddings already exist. Only do so for dataset iNaturalist as the others have varying train/val splits
        if self.ds_name == "inaturalist":
            if os.path.exists(os.path.join(self.root, f"{self.ds_name}_train_UMAP_embeddings.pkl")):
                self.umap_embeddings = pickle.load(
                    open(os.path.join(self.root, f"{self.ds_name}_train_UMAP_embeddings.pkl"), "rb")
                )
                self.df = pickle.load(open(os.path.join(self.root, f"{self.ds_name}_UMAP_df.pkl"), "rb"))
                print("Embeddings already exist")
            else: 
                self.construct_embeddings()
                pickle.dump(self.umap_embeddings, open(os.path.join(self.root, f"{self.ds_name}_train_UMAP_embeddings.pkl"), "wb"))
                pickle.dump(self.df, open(os.path.join(self.root, f"{self.ds_name}_UMAP_df.pkl"), "wb"))
        else:
            self.construct_embeddings()


    def construct_embeddings(self):
        # Create embeddings for the whole training set
        self.train_embeddings = self._make_embedding_for_dataloader(
            self.full_trainloader
        )
        cols = [f"out_{i}" for i in range(self.train_embeddings.shape[1])]
        self.train_labels = [y for x, y in self.full_trainloader.dataset]

        # create UMAP plot
        time_start = time.time()
        print('Starting PCA and UMAP fit and transform.')
        # transform to 50 pca components
        pca_50 = PCA(n_components=50)
        self.train_embeddings = pca_50.fit_transform(self.train_embeddings)
        print('PCA done in {} seconds.'.format(time.time() - time_start))
        time_start = time.time()

        self.umap_embeddings = self.umap.fit_transform(X=self.train_embeddings, y=self.train_labels)
        print("UMAP done! Time elapsed: {} seconds".format(time.time() - time_start))

        print('shape: ', self.umap_embeddings.shape)
        print('type of embeddings: ', type(self.umap_embeddings))
        print('type of labels: ', type(self.train_labels))
        print('type of first label in labels: ', type(self.train_labels[0]))
        print('type of first embedding: ', type(self.umap_embeddings[0]))
        self.df = pd.DataFrame(self.umap_embeddings, columns=["umap-2d-x", "umap-2d-y"])

        # train_labels is a list of tensors. Make it a list of ints
        self.df["LABEL"] = [int(x) for x in self.train_labels]
        self.df["umap-2d-x"] = self.umap_embeddings[:, 0]
        self.df["umap-2d-y"] = self.umap_embeddings[:, 1]

        

    def make_plot(self, epoch, selected_indices=None):
        # reset all the selected indices to False
        # if a row is selected, mark it as selected in the dataframe
        if selected_indices is not None:
            self.df["selected_indices"] = False
            self.df.loc[selected_indices, "selected_indices"] = True

        # table = wandb.Table(columns=self.df.columns.tolist(), data=self.df.values)
        # wandb.log({"umap_data": table}, step=epoch)

        # plot the umap plot
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="umap-2d-x",
            y="umap-2d-y",
            hue="LABEL",
            palette=sns.color_palette("hls", len(self.df["LABEL"].unique())),
            data=self.df,
            legend="full",
            alpha=0.7,
        )

        # visualize every selected point
        if selected_indices is not None:
            plt.scatter(
                self.df["umap-2d-x"][selected_indices],
                self.df["umap-2d-y"][selected_indices],
                marker="*",
                color="black",
                s=80,
            )
        plt.title("UMAP plot")

        # random file name to save the plot
        file_name = f"UMAP_{time.time()}.png"
        plt.savefig(file_name)
        # wandb.log({"UMAP_plot": wandb.Image(file_name)}, step=epoch)
        plt.show()
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

    # make a dataset and a dataloader
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    tp = UMAPPlotter(dataloader, dataloader, dataloader, None, 'cpu', '../data', 'cifar')
    tp.make_plot(0, [1, 2, 3, 4, 5])
