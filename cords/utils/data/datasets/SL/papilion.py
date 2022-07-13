import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import gdown
import numpy as np
import pandas as pd
import PIL
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter


class PapilionDataset(Dataset):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=True
    ):
        """Load the dataset.
        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(
            root + "/papilion"
        )  # Replace ~ by the complete dir
        self._train = train
        self._transform = transform
        self._target_transform = target_transform
        self._fraction = 0.33

        if self._checkIntegrity():
            print("Dataset files present and verified")
        else:
            print("Dataset files are missing, downloading...")
            self._download()

        # Load train and test data from pickle files
        if self._train:
            with open(os.path.join(self._root + "/train.pkl"), "rb") as f:
                self._train_data, self._train_labels = pickle.load(f)
                self.data = self._train_data
                self.targets = self._train_labels
        else:
            with open(os.path.join(self._root + "/test.pkl"), "rb") as f:
                self._test_data, self._test_labels = pickle.load(f)
                self.data = self._test_data
                self.targets = self._test_labels

        # targets should be a tensor
        self.targets = torch.tensor(self.targets)


    def __getitem__(self, index):
        """
        Args:
            index, int: Index.
        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        # Doing this so that it is consistent with all other datasets.
        image = PIL.Image.fromarray(image)

        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            target = self._target_transform(target)

        return image, target

    def __len__(self):
        """Length of the dataset.
        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _checkIntegrity(self):
        """Check if the files are present and correct.
        Returns:
            bool: True if the files are present and correct, False otherwise.
        """
        return os.path.isfile(
            os.path.join(self._root + "/train.pkl")
        ) and os.path.isfile(os.path.join(self._root + "/test.pkl"))

    def _download(self):
        """Download the dataset."""
        raw_path = os.path.join(self._root, "raw")
        if not os.path.isdir(raw_path):
            print("making directory: " + raw_path)
            os.makedirs(raw_path)

        # Download the data from google drive
        # url https://drive.google.com/file/d/1ORfn9gl4dEOJFPvWjqGJ9T1ZT7vFZKKd/view?usp=sharing
        if not os.path.isfile(os.path.join(raw_path + "/papilion.zip")):
            gdown.download(
                "https://drive.google.com/uc?id=1ORfn9gl4dEOJFPvWjqGJ9T1ZT7vFZKKd",
                os.path.join(raw_path, "papilion.zip"),
                quiet=False,
            )
        self._processData()

    def _processData(self):
        # process the downloaded raw data into train/test data and save as pickle files
        # unzip the zip file
        import zipfile

        # unzip and allow file overwrite
        print("unzipping... ", os.path.join(self._root, "raw/papilion.zip"))
        with zipfile.ZipFile(
            os.path.join(self._root + "/raw/papilion.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(self._root + "/raw")
        print("unzipping done")

        # metadata file
        # this csv file has the following columns: catalogNumber,basisOfRecord,class,collectionCode,continent,country,city,dateIdentified,latitudeDecimal,longitudeDecimal,family,genus,identifiedBy,individualCount,infraspecificEpithet,island,kingdom,lifeStage,locality,occurrenceID,order,phylum,preparations,recordedBy,scientificName,authorshipVerbatim,sex,specificEpithet,provinceState,subgenus,taxonRank,remarks,typeStatus,depth,altitudeUnifOfMeasurement,taxonRank.1,associatedMedia,verbatimCoordinates,verbatimEventDate,higherClassification,informationWithheld,verbatimCoordinates.1,eventDate,nomenclaturalCode,geodeticDatum,uid,crop_exists,class_infra_species,class_species,class_genus
        metadata_file = os.path.join(self._root, "raw/images.csv")
        images_path = os.path.join(self._root, "raw/images_crop/")
        label_column = "class_infra_species"
        image_id_column = "uid"

        all_data = []
        all_labels = []

        # read the csv file with pandas such that we can use column names
        df = pd.read_csv(metadata_file)
        column_names = df.columns.tolist()
        label_column_index = column_names.index(label_column)
        image_id_column_index = column_names.index(image_id_column)

        missing = 0
        # iterate over the rows of the csv file
        for index, row in df.iterrows():
            image_id = row[image_id_column_index]
            label = row[label_column_index]
            image_path = os.path.join(images_path, image_id + ".jpg")
            if os.path.isfile(image_path):
                image = PIL.Image.open(image_path)
                image_np = np.array(image)
                image.close()
                all_data.append(image_np)
                all_labels.append(label)
            else:
                missing += 1
        # print("missing: " + str(missing), "set size: ", len(all_data))

        # the dataset is very imbalanced so we need to do stratified sampling
        # to balance the classes
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=self._fraction, random_state=42
        )

        all_data = np.array(all_data)
        all_labels = np.array(all_labels)

        for train_index, test_index in sss.split(all_data, all_labels):
            self._train_data = all_data[train_index]
            self._train_labels = all_labels[train_index]
            self._test_data = all_data[test_index]
            self._test_labels = all_labels[test_index]
            break

        # save the data as pickle files
        pickle.dump(
            (self._train_data, self._train_labels),
            open(os.path.join(self._root, "train.pkl"), "wb"),
        )
        pickle.dump(
            (self._test_data, self._test_labels),
            open(os.path.join(self._root, "test.pkl"), "wb"),
        )

    def _visualize(self):
        # visualize the class distribution
        import matplotlib.pyplot as plt
        import seaborn as sns

        class_counts = Counter(self._train_labels)
        # we need to convert the x-axis to numeric values
        sns.barplot(
            x=list(range(len(class_counts))),
            y=list(class_counts.values()),
            palette="husl",
            orient="v",
        )
        plt.show()
        class_counts = Counter(self._test_labels)
        sns.barplot(
            x=list(range(len(class_counts))),
            y=list(class_counts.values()),
            palette="husl",
            orient="v",
        )
        plt.show()

        print(
            "train data mean resolution: ",
            np.mean(self._train_data.shape[1:]),
            " test data mean resolution: ",
            np.mean(self._test_data.shape[1:]),
        )


if __name__ == "__main__":
    PapilionDataset(root="~/data", train=True)
