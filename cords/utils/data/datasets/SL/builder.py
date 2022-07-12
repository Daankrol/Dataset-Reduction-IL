import numpy as np
import os

from dotmap import DotMap

from cords.utils.data.datasets.SL.custom_dataset_selcon import (
    CustomDataset_WithId_SELCON,
)
import torchvision
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split, TensorDataset, Subset
from torchvision import transforms
import PIL.Image as Image
import PIL
from sklearn.datasets import load_boston
from cords.utils.data.data_utils import *
import re
import pandas as pd
import torch
import pickle
from cords.utils.data.data_utils import WeightedSubset
from sklearn.model_selection import StratifiedShuffleSplit
from cords.utils.data.datasets.SL.papilion import PapilionDataset


class standard_scaling:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit_transform(self, data):
        self.std = np.std(data, axis=0)
        self.mean = np.mean(data, axis=0)
        transformed_data = np.subtract(data, self.mean)
        transformed_data = np.divide(transformed_data, self.std)
        return transformed_data

    def transform(self, data):
        transformed_data = np.subtract(data, self.mean)
        transformed_data = np.divide(transformed_data, self.std)
        return transformed_data


def clean_data(sentence, type=0, TREC=False):
    #     From yoonkim: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    if type == 0:
        """
        Tokenization for SST
        """
        sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
        sentence = re.sub(r"\s{2,}", " ", sentence)
        return sentence.strip().lower()
    elif type == 1:
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
        sentence = re.sub(r"\'s", " 's", sentence)
        sentence = re.sub(r"\'ve", " 've", sentence)
        sentence = re.sub(r"n\'t", " n't", sentence)
        sentence = re.sub(r"\'re", " 're", sentence)
        sentence = re.sub(r"\'d", " 'd", sentence)
        sentence = re.sub(r"\'ll", " 'll", sentence)
        sentence = re.sub(r",", " , ", sentence)
        sentence = re.sub(r"!", " ! ", sentence)
        sentence = re.sub(r"\(", " \( ", sentence)
        sentence = re.sub(r"\)", " \) ", sentence)
        sentence = re.sub(r"\?", " \? ", sentence)
        sentence = re.sub(r"\s{2,}", " ", sentence)
        return sentence.strip() if TREC else sentence.strip().lower()
        # if we are using glove uncased, keep TREC = False even for trec6 dataset
    else:
        return sentence


def get_class(sentiment, num_classes):
    # Return a label based on the sentiment value
    return int(sentiment * (num_classes - 0.001))


def loadGloveModel(gloveFile):
    glove = pd.read_csv(
        gloveFile,
        sep=" ",
        header=None,
        encoding="utf-8",
        index_col=0,
        na_values=None,
        keep_default_na=False,
        quoting=3,
    )
    return glove  # (word, embedding), 400k*dim


class SSTDataset(Dataset):
    label_tmp = None

    def __init__(
        self, path_to_dataset, name, num_classes, wordvec_dim, wordvec, device="cpu"
    ):
        """SST dataset

        Args:
            path_to_dataset (str): path_to_dataset
            name (str): train, dev or test
            num_classes (int): 2 or 5
            wordvec_dim (int): Dimension of word embedding
            wordvec (array): word embedding
            device (str, optional): torch.device. Defaults to 'cpu'.
        """
        phrase_ids = pd.read_csv(
            path_to_dataset + "phrase_ids." + name + ".txt",
            header=None,
            encoding="utf-8",
            dtype=int,
        )
        phrase_ids = set(np.array(phrase_ids).squeeze())  # phrase_id in this dataset
        self.num_classes = num_classes
        phrase_dict = {}  # {id->phrase}

        if SSTDataset.label_tmp is None:
            # Read label/sentiment first
            # Share 1 array on train/dev/test set. No need to do this 3 times.
            SSTDataset.label_tmp = pd.read_csv(
                path_to_dataset + "sentiment_labels.txt",
                sep="|",
                dtype={"phrase ids": int, "sentiment values": float},
            )
            SSTDataset.label_tmp = np.array(SSTDataset.label_tmp)[
                :, 1:
            ]  # sentiment value

        with open(path_to_dataset + "dictionary.txt", "r", encoding="utf-8") as f:
            i = 0
            for line in f:
                phrase, phrase_id = line.strip().split("|")
                if int(phrase_id) in phrase_ids:  # phrase in this dataset
                    phrase = clean_data(phrase)  # preprocessing
                    phrase_dict[int(phrase_id)] = phrase
                    i += 1
        f.close()

        self.phrase_vec = []  # word index in glove
        # label of each sentence
        self.labels = torch.zeros((len(phrase_dict),), dtype=torch.long)
        missing_count = 0
        for i, (idx, p) in enumerate(phrase_dict.items()):
            tmp1 = []
            for w in p.split(" "):
                try:
                    tmp1.append(wordvec.index.get_loc(w))
                except KeyError:
                    missing_count += 1

            self.phrase_vec.append(torch.tensor(tmp1, dtype=torch.long))
            self.labels[i] = get_class(SSTDataset.label_tmp[idx], self.num_classes)

        # print(missing_count)

    def __getitem__(self, index):
        return self.phrase_vec[index], self.labels[index]

    def __len__(self):
        return len(self.phrase_vec)


class CUB200(torch.utils.data.Dataset):
    """CUB200 dataset.
    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _target_transform, callable: A function/transform that takes in the
            target and transforms it.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """

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
            root + "/cub200"
        )  # Replace ~ by the complete dir
        self._train = train
        self._transform = transform
        self._target_transform = target_transform

        if self._checkIntegrity():
            print("Files already downloaded and verified.")
        elif download:
            # url = (
            #     "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/"
            #     "CUB_200_2011.tgz"
            # )
            print(
                "Download CUB200 from https://drive.google.com/uc?id=1U655cnOmqRZHEindgJgIQ49Cm-Kgro8d"
                + " and put it in "
                + self._root
                + "/raw/CUB_200_2011/"
            )
            # exit()
            # self._download(url)
            self._extract()
        else:
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )

        # Now load the picked data.
        if self._train:
            self._train_data, self._train_labels = pickle.load(
                open(os.path.join(self._root, "processed/train.pkl"), "rb")
            )
            assert len(self._train_data) == 5994 and len(self._train_labels) == 5994
            self.data = self._train_data
            self.targets = self._train_labels
        else:
            self._test_data, self._test_labels = pickle.load(
                open(os.path.join(self._root, "processed/test.pkl"), "rb")
            )
            assert len(self._test_data) == 5794 and len(self._test_labels) == 5794
            self.data = self._test_data
            self.targets = self._test_labels

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
        # image = PIL.Image.fromarray(image)

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
        """Check whether we have already processed the data.
        Returns:
            flag, bool: True if we have already processed the data.
        """
        return os.path.isfile(
            os.path.join(self._root, "processed/train.pkl")
        ) and os.path.isfile(os.path.join(self._root, "processed/test.pkl"))

    def _download(self, url):
        """Download and uncompress the tar.gz file from a given URL.
        Args:
            url, str: URL to be downloaded.
        """
        import six.moves
        import tarfile

        raw_path = os.path.join(self._root, "raw")
        processed_path = os.path.join(self._root, "processed")
        if not os.path.isdir(raw_path):
            os.mkdir(raw_path, mode=0o775)
        if not os.path.isdir(processed_path):
            os.mkdir(processed_path, mode=0o775)

        # Downloads file.
        fpath = os.path.join(self._root, "raw/CUB_200_2011.tgz")
        try:
            print("Downloading " + url + " to " + fpath)
            six.moves.urllib.request.urlretrieve(url, fpath)
        except six.moves.urllib.error.URLError:
            if url[:5] == "https:":
                self._url = self._url.replace("https:", "http:")
                print("Failed download. Trying https -> http instead.")
                print("Downloading " + url + " to " + fpath)
                six.moves.urllib.request.urlretrieve(url, fpath)

        # Extract file.
        cwd = os.getcwd()
        tar = tarfile.open(fpath, "r:gz")
        os.chdir(os.path.join(self._root, "raw"))
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def _extract(self):
        """Prepare the data for train/test split and save onto disk."""
        image_path = os.path.join(self._root, "raw/CUB_200_2011/images/")
        print("Extracting images from " + image_path)
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(
            os.path.join(self._root, "raw/CUB_200_2011/images.txt"), dtype=str
        )
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(
            os.path.join(self._root, "raw/CUB_200_2011/train_test_split.txt"), dtype=int
        )

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for id_ in range(id2name.shape[0]):
            image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
            label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

            # Convert gray scale image to RGB image.
            if image.getbands()[0] == "L":
                image = image.convert("RGB")
            image_np = np.array(image)
            image.close()

            if id2train[id_, 1] == 1:
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_data.append(image_np)
                test_labels.append(label)

        pickle.dump(
            (train_data, train_labels),
            open(os.path.join(self._root, "processed/train.pkl"), "wb"),
        )
        pickle.dump(
            (test_data, test_labels),
            open(os.path.join(self._root, "processed/test.pkl"), "wb"),
        )


class Trec6Dataset(Dataset):
    def __init__(
        self, data_path, cls_to_num, num_classes, wordvec_dim, wordvec, device="cpu"
    ):
        self.phrase_vec = []
        self.labels = []

        missing_count = 0
        with open(data_path, "r", encoding="latin1") as f:
            for line in f:
                label = cls_to_num[line.split()[0].split(":")[0]]
                sentence = clean_data(" ".join(line.split(":")[1:]), 1, False)

                tmp1 = []
                for w in sentence.split(" "):
                    try:
                        tmp1.append(wordvec.index.get_loc(w))
                    except KeyError:
                        missing_count += 1

                self.phrase_vec.append(torch.tensor(tmp1, dtype=torch.long))
                self.labels.append(label)

    def __getitem__(self, index):
        return self.phrase_vec[index], self.labels[index]

    def __len__(self):
        return len(self.phrase_vec)


class GlueDataset(Dataset):
    def __init__(
        self,
        glue_dataset,
        sentence_str,
        label_str,
        clean_type,
        num_classes,
        wordvec_dim,
        wordvec,
        device="cpu",
    ):
        self.len = glue_dataset.__len__()
        self.phrase_vec = []  # word index in glove
        # label of each sentence
        self.labels = torch.zeros((self.len,), dtype=torch.long)
        missing_count = 0
        for i, p in enumerate(glue_dataset):
            tmp1 = []
            for w in clean_data(p[sentence_str], clean_type, False).split(
                " "
            ):  # False since glove used is uncased
                try:
                    tmp1.append(wordvec.index.get_loc(w))
                except KeyError:
                    missing_count += 1

            self.phrase_vec.append(torch.tensor(tmp1, dtype=torch.long))
            self.labels[i] = p[label_str]

    def __getitem__(self, index):
        return self.phrase_vec[index], self.labels[index]

    def __len__(self):
        return self.len


## Custom PyTorch Dataset Class wrapper
class CustomDataset(Dataset):
    def __init__(self, data, target, device=None, transform=None, isreg=False):
        self.transform = transform
        self.isreg = isreg
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.float().to(device)
            if isreg:
                self.targets = target.float().to(device)
            else:
                self.targets = target.long().to(device)

        else:
            self.data = data.float()
            if isreg:
                self.targets = target.float()
            else:
                self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return (sample_data, label)  # .astype('float32')
        # if self.isreg:
        #     return (sample_data, label, idx)
        # else:


class CustomDataset_WithId(Dataset):
    def __init__(self, data, target, device=None, transform=None, isreg=False):
        self.transform = transform
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.float().to(device)
            if isreg:
                self.targets = target.float().to(device)
            else:
                self.targets = target.long().to(device)

        else:
            self.data = data.float()
            if isreg:
                self.targets = target.float()
            else:
                self.targets = target.long()
        self.X = self.data
        self.Y = self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return sample_data, label, idx  # .astype('float32')


## Utility function to load datasets from libsvm datasets
def csv_file_load(path, dim, save_data=False):
    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i for i in line.strip().split(",")]
            target.append(
                int(float(temp[-1]))
            )  # Class Number. # Not assumed to be in (0, K-1)
            temp_data = [0] * dim
            count = 0
            for i in temp[:-1]:
                # ind, val = i.split(':')
                temp_data[count] = float(i)
                count += 1
            data.append(temp_data)
            line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + ".data.npy"
        target_np_path = path + ".label.npy"
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)


def libsvm_file_load(path, dim, save_data=False):
    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i for i in line.strip().split(" ")]
            target.append(
                int(float(temp[0]))
            )  # Class Number. # Not assumed to be in (0, K-1)
            temp_data = [0] * dim

            for i in temp[1:]:
                ind, val = i.split(":")
                temp_data[int(ind) - 1] = float(val)
            data.append(temp_data)
            line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + ".data.npy"
        target_np_path = path + ".label.npy"
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)


def clean_lawschool_full(path):
    df = pd.read_csv(path)
    df = df.dropna()
    # remove y from df
    y = df["ugpa"]
    y = y / 4
    df = df.drop("ugpa", axis=1)
    # convert gender variables to 0,1
    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    # add bar1 back to the feature set
    df_bar = df["bar1"]
    df = df.drop("bar1", axis=1)
    df["bar1"] = [int(grade == "P") for grade in df_bar]
    # df['race'] = [int(race == 7.0) for race in df['race']]
    # a = df['race']
    return df.to_numpy(), y.to_numpy()


def census_load(path, dim, save_data=False):
    enum = enumerate(
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ]
    )
    workclass = dict((j, i) for i, j in enum)

    enum = enumerate(
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ]
    )
    education = dict((j, i) for i, j in enum)

    enum = enumerate(
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ]
    )
    marital_status = dict((j, i) for i, j in enum)

    enum = enumerate(
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ]
    )
    occupation = dict((j, i) for i, j in enum)

    enum = enumerate(
        ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
    )
    relationship = dict((j, i) for i, j in enum)

    enum = enumerate(
        ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    )
    race = dict((j, i) for i, j in enum)

    sex = {"Female": 0, "Male": 1}

    enum = enumerate(
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ]
    )
    native_country = dict((j, i) for i, j in enum)

    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i.strip() for i in line.strip().split(",")]

            if "?" in temp or len(temp) == 1:
                line = fp.readline()
                continue

            if temp[-1].strip() == "<=50K" or temp[-1].strip() == "<=50K.":
                target.append(0)
            else:
                target.append(1)

            temp_data = [0] * dim
            count = 0
            # print(temp)

            for i in temp[:-1]:

                if count == 1:
                    temp_data[count] = workclass[i.strip()]
                elif count == 3:
                    temp_data[count] = education[i.strip()]
                elif count == 5:
                    temp_data[count] = marital_status[i.strip()]
                elif count == 6:
                    temp_data[count] = occupation[i.strip()]
                elif count == 7:
                    temp_data[count] = relationship[i.strip()]
                elif count == 8:
                    temp_data[count] = race[i.strip()]
                elif count == 9:
                    temp_data[count] = sex[i.strip()]
                elif count == 13:
                    temp_data[count] = native_country[i.strip()]
                else:
                    temp_data[count] = float(i)
                temp_data[count] = float(temp_data[count])
                count += 1

            data.append(temp_data)
            line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + ".data.npy"
        target_np_path = path + ".label.npy"
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)


def create_imbalance(x_trn, y_trn, x_val, y_val, x_tst, y_tst, num_cls, ratio):
    np.random.seed(42)
    samples_per_class = np.zeros(num_cls)
    val_samples_per_class = np.zeros(num_cls)
    tst_samples_per_class = np.zeros(num_cls)
    for i in range(num_cls):
        samples_per_class[i] = len(np.where(y_trn == i)[0])
        val_samples_per_class[i] = len(np.where(y_val == i)[0])
        tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
    min_samples = int(np.min(samples_per_class) * 0.1)
    selected_classes = np.random.choice(
        np.arange(num_cls), size=int(ratio * num_cls), replace=False
    )
    for i in range(num_cls):
        if i == 0:
            if i in selected_classes:
                subset_idxs = np.random.choice(
                    np.where(y_trn == i)[0], size=min_samples, replace=False
                )
            else:
                subset_idxs = np.where(y_trn == i)[0]
            x_trn_new = x_trn[subset_idxs]
            y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
        else:
            if i in selected_classes:
                subset_idxs = np.random.choice(
                    np.where(y_trn == i)[0], size=min_samples, replace=False
                )
            else:
                subset_idxs = np.where(y_trn == i)[0]
            x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
            y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))
    max_samples = int(np.max(val_samples_per_class))
    for i in range(num_cls):
        y_class = np.where(y_val == i)[0]
        if i == 0:
            subset_ids = np.random.choice(
                y_class, size=max_samples - y_class.shape[0], replace=True
            )
            x_val_new = np.row_stack((x_val, x_val[subset_ids]))
            y_val_new = np.row_stack(
                (y_val.reshape(-1, 1), y_val[subset_ids].reshape(-1, 1))
            )
        else:
            subset_ids = np.random.choice(
                y_class, size=max_samples - y_class.shape[0], replace=True
            )
            x_val_new = np.row_stack((x_val, x_val_new, x_val[subset_ids]))
            y_val_new = np.row_stack(
                (y_val.reshape(-1, 1), y_val_new, y_val[subset_ids].reshape(-1, 1))
            )
    max_samples = int(np.max(tst_samples_per_class))
    for i in range(num_cls):
        y_class = np.where(y_tst == i)[0]
        if i == 0:
            subset_ids = np.random.choice(
                y_class, size=max_samples - y_class.shape[0], replace=True
            )
            x_tst_new = np.row_stack((x_tst, x_tst[subset_ids]))
            y_tst_new = np.row_stack(
                (y_tst.reshape(-1, 1), y_tst[subset_ids].reshape(-1, 1))
            )
        else:
            subset_ids = np.random.choice(
                y_class, size=max_samples - y_class.shape[0], replace=True
            )
            x_tst_new = np.row_stack((x_tst, x_tst_new, x_tst[subset_ids]))
            y_tst_new = np.row_stack(
                (y_tst.reshape(-1, 1), y_tst_new, y_tst[subset_ids].reshape(-1, 1))
            )

    return (
        x_trn_new,
        y_trn_new.reshape(-1),
        x_val_new,
        y_val_new.reshape(-1),
        x_tst_new,
        y_tst_new.reshape(-1),
    )


def create_noisy(y_trn, num_cls, noise_ratio=0.8):
    noise_size = int(len(y_trn) * noise_ratio)
    noise_indices = np.random.choice(
        np.arange(len(y_trn)), size=noise_size, replace=False
    )
    y_trn[noise_indices] = np.random.choice(
        np.arange(num_cls), size=noise_size, replace=True
    )
    return y_trn


def gen_dataset(datadir, dset_name, feature, isnumpy=False, **kwargs):
    """
    Generate train, val, and test datasets for supervised learning setting.

    Parameters
    --------
    datadir: str
        Dataset directory in which the data is present or needs to be downloaded.
    dset_name: str
        dataset name, ['cifar10', 'cifar100', 'svhn', 'stl10', 'cub200']
    feature: str
        if 'classimb', generate datasets wth class imbalance
            - Needs keyword argument 'classimb_ratio'
        elif 'noise', generate datasets with label noise
        otherwise, generate standard datasets
    isnumpy: bool
        if True, return datasets in numpy format instead of tensor format
    """

    if feature == "classimb":
        if "classimb_ratio" in kwargs:
            pass
        else:
            raise KeyError("Specify a classimbratio value in the config file")

    if dset_name == "dna":
        np.random.seed(42)
        trn_file = os.path.join(datadir, "dna.scale.trn")
        val_file = os.path.join(datadir, "dna.scale.val")
        tst_file = os.path.join(datadir, "dna.scale.tst")
        data_dims = 180
        num_cls = 3
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)

        y_trn -= 1  # First Class should be zero
        y_val -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )
        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))
        return fullset, valset, testset, num_cls

    elif dset_name == "boston":
        num_cls = 1
        x_trn, y_trn = load_boston(return_X_y=True)

        # create train and test indices
        # train, test = train_test_split(list(range(X.shape[0])), test_size=.3)
        x_trn, x_tst, y_trn, y_tst = train_test_split(
            x_trn, y_trn, test_size=0.2, random_state=42
        )
        x_trn, x_val, y_trn, y_val = train_test_split(
            x_trn, y_trn, test_size=0.1, random_state=42
        )
        scaler = standard_scaling()
        x_trn = scaler.fit_transform(x_trn)
        x_val = scaler.transform(x_val)
        x_tst = scaler.transform(x_tst)
        y_trn = y_trn.reshape((-1, 1))
        y_val = y_val.reshape((-1, 1))
        y_tst = y_tst.reshape((-1, 1))
        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)
        else:
            fullset = CustomDataset(
                torch.from_numpy(x_trn), torch.from_numpy(y_trn), isreg=True
            )
            valset = CustomDataset(
                torch.from_numpy(x_val), torch.from_numpy(y_val), isreg=True
            )
            testset = CustomDataset(
                torch.from_numpy(x_tst), torch.from_numpy(y_tst), isreg=True
            )
        return fullset, valset, testset, num_cls

    elif dset_name in ["Community_Crime", "LawSchool_selcon"]:
        if dset_name == "Community_Crime":
            x_trn, y_trn = clean_communities_full(
                os.path.join(datadir, "communities.scv")
            )
        elif dset_name == "LawSchool_selcon":
            x_trn, y_trn = clean_lawschool_full(os.path.join(datadir, "lawschool.csv"))
        else:
            raise NotImplementedError

        fullset = (x_trn, y_trn)
        data_dims = x_trn.shape[1]
        device = "cpu"

        (
            x_trn,
            y_trn,
            x_val_list,
            y_val_list,
            val_classes,
            x_tst_list,
            y_tst_list,
            tst_classes,
        ) = get_slices(dset_name, fullset[0], fullset[1], device, 3)

        assert val_classes == tst_classes

        trainset = CustomDataset_WithId_SELCON(
            torch.from_numpy(x_trn).float().to(device),
            torch.from_numpy(y_trn).float().to(device),
        )
        valset = CustomDataset_WithId_SELCON(
            torch.cat(x_val_list, dim=0), torch.cat(y_val_list, dim=0)
        )
        testset = CustomDataset_WithId_SELCON(
            torch.cat(x_tst_list, dim=0), torch.cat(y_tst_list, dim=0)
        )

        return trainset, valset, testset, val_classes

    elif dset_name in ["cadata", "abalone", "cpusmall", "LawSchool"]:

        if dset_name == "cadata":
            trn_file = os.path.join(datadir, "cadata.txt")
            x_trn, y_trn = libsvm_file_load(trn_file, dim=8)

        elif dset_name == "abalone":
            trn_file = os.path.join(datadir, "abalone_scale.txt")
            x_trn, y_trn = libsvm_file_load(trn_file, 8)

        elif dset_name == "cpusmall":
            trn_file = os.path.join(datadir, "cpusmall_scale.txt")
            x_trn, y_trn = libsvm_file_load(trn_file, 12)

        elif dset_name == "LawSchool":
            x_trn, y_trn = clean_lawschool_full(os.path.join(datadir, "lawschool.csv"))

        # create train and test indices
        # train, test = train_test_split(list(range(X.shape[0])), test_size=.3)
        x_trn, x_tst, y_trn, y_tst = train_test_split(
            x_trn, y_trn, test_size=0.2, random_state=42
        )
        x_trn, x_val, y_trn, y_val = train_test_split(
            x_trn, y_trn, test_size=0.1, random_state=42
        )

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        sc_l = StandardScaler()
        y_trn = np.reshape(sc_l.fit_transform(np.reshape(y_trn, (-1, 1))), (-1))
        y_val = np.reshape(sc_l.fit_transform(np.reshape(y_val, (-1, 1))), (-1))
        y_tst = np.reshape(sc_l.fit_transform(np.reshape(y_tst, (-1, 1))), (-1))

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset_WithId(
                torch.from_numpy(x_trn), torch.from_numpy(y_trn), isreg=True
            )
            valset = CustomDataset_WithId(
                torch.from_numpy(x_val), torch.from_numpy(y_val), isreg=True
            )
            testset = CustomDataset_WithId(
                torch.from_numpy(x_tst), torch.from_numpy(y_tst), isreg=True
            )

        return fullset, valset, testset, 1

    elif dset_name == "MSD":

        trn_file = os.path.join(datadir, "YearPredictionMSD")
        x_trn, y_trn = libsvm_file_load(trn_file, 90)

        tst_file = os.path.join(datadir, "YearPredictionMSD.t")
        x_tst, y_tst = libsvm_file_load(tst_file, 90)
        x_trn, x_val, y_trn, y_val = train_test_split(
            x_trn, y_trn, test_size=0.005, random_state=42
        )

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        sc_l = StandardScaler()
        y_trn = np.reshape(sc_l.fit_transform(np.reshape(y_trn, (-1, 1))), (-1))
        y_val = np.reshape(sc_l.fit_transform(np.reshape(y_val, (-1, 1))), (-1))
        y_tst = np.reshape(sc_l.fit_transform(np.reshape(y_tst, (-1, 1))), (-1))

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(
                torch.from_numpy(x_trn), torch.from_numpy(y_trn), if_reg=True
            )
            valset = CustomDataset(
                torch.from_numpy(x_val), torch.from_numpy(y_val), if_reg=True
            )
            testset = CustomDataset(
                torch.from_numpy(x_tst), torch.from_numpy(y_tst), if_reg=True
            )

        return fullset, valset, testset, 1

    elif dset_name == "adult":
        trn_file = os.path.join(datadir, "a9a.trn")
        tst_file = os.path.join(datadir, "a9a.tst")
        data_dims = 123
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)

        y_trn[y_trn < 0] = 0
        y_tst[y_tst < 0] = 0

        x_trn, x_val, y_trn, y_val = train_test_split(
            x_trn, y_trn, test_size=0.1, random_state=42
        )

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )
        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "connect_4":
        trn_file = os.path.join(datadir, "connect_4.trn")

        data_dims = 126
        num_cls = 3

        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        # The class labels are (-1,0,1). Make them to (0,1,2)
        y_trn[y_trn < 0] = 2

        x_trn, x_tst, y_trn, y_tst = train_test_split(
            x_trn, y_trn, test_size=0.1, random_state=42
        )
        x_trn, x_val, y_trn, y_val = train_test_split(
            x_trn, y_trn, test_size=0.1, random_state=42
        )

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )
        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "letter":
        trn_file = os.path.join(datadir, "letter.scale.trn")
        val_file = os.path.join(datadir, "letter.scale.val")
        tst_file = os.path.join(datadir, "letter.scale.tst")
        data_dims = 16
        num_cls = 26
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )

        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "satimage":
        np.random.seed(42)
        trn_file = os.path.join(datadir, "satimage.scale.trn")
        val_file = os.path.join(datadir, "satimage.scale.val")
        tst_file = os.path.join(datadir, "satimage.scale.tst")
        data_dims = 36
        num_cls = 6

        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)

        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )

        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "svmguide1":
        np.random.seed(42)
        trn_file = os.path.join(datadir, "svmguide1.trn_full")
        tst_file = os.path.join(datadir, "svmguide1.tst")
        data_dims = 4
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)

        x_trn, x_val, y_trn, y_val = train_test_split(
            x_trn, y_trn, test_size=0.1, random_state=42
        )

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )

        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "usps":
        np.random.seed(42)
        trn_file = os.path.join(datadir, "usps.trn_full")
        tst_file = os.path.join(datadir, "usps.tst")
        data_dims = 256
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero

        x_trn, x_val, y_trn, y_val = train_test_split(
            x_trn, y_trn, test_size=0.1, random_state=42
        )
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )

        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "ijcnn1":
        np.random.seed(42)
        trn_file = os.path.join(datadir, "ijcnn1.trn")
        val_file = os.path.join(datadir, "ijcnn1.val")
        tst_file = os.path.join(datadir, "ijcnn1.tst")
        data_dims = 22
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)

        # The class labels are (-1,1). Make them to (0,1)
        y_trn[y_trn < 0] = 0
        y_val[y_val < 0] = 0
        y_tst[y_tst < 0] = 0

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )

        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name == "sklearn-digits":
        np.random.seed(42)
        data, target = datasets.load_digits(return_X_y=True)
        # Test data is 10%
        x_trn, x_tst, y_trn, y_tst = train_test_split(
            data, target, test_size=0.1, random_state=42
        )

        x_trn, x_val, y_trn, y_val = train_test_split(
            x_trn, y_trn, test_size=0.1, random_state=42
        )
        num_cls = 10
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )

        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, num_cls

    elif dset_name in [
        "prior_shift_large_linsep_4",
        "conv_shift_large_linsep_4",
        "red_large_linsep_4",
        "expand_large_linsep_4",
        "shrink_large_linsep_4",
        "red_conv_shift_large_linsep_4",
        "linsep_4",
        "large_linsep_4",
    ]:

        np.random.seed(42)
        trn_file = os.path.join(datadir, dset_name + ".trn")
        val_file = os.path.join(datadir, dset_name + ".val")
        tst_file = os.path.join(datadir, dset_name + ".tst")
        data_dims = 2
        num_cls = 4
        x_trn, y_trn = csv_file_load(trn_file, dim=data_dims)
        x_val, y_val = csv_file_load(val_file, dim=data_dims)
        x_tst, y_tst = csv_file_load(tst_file, dim=data_dims)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )

        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, num_cls

    elif dset_name in [
        "prior_shift_clf_2",
        "prior_shift_gauss_2",
        "conv_shift_clf_2",
        "conv_shift_gauss_2",
        "gauss_2",
        "clf_2",
        "linsep",
    ]:
        np.random.seed(42)
        trn_file = os.path.join(datadir, dset_name + ".trn")
        val_file = os.path.join(datadir, dset_name + ".val")
        tst_file = os.path.join(datadir, dset_name + ".tst")
        data_dims = 2
        num_cls = 2
        x_trn, y_trn = csv_file_load(trn_file, dim=data_dims)
        x_val, y_val = csv_file_load(val_file, dim=data_dims)
        x_tst, y_tst = csv_file_load(tst_file, dim=data_dims)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )

        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, num_cls

    elif dset_name == "covertype":
        np.random.seed(42)
        trn_file = os.path.join(datadir, "covtype.data")
        data_dims = 54
        num_cls = 7
        x_trn, y_trn = csv_file_load(trn_file, dim=data_dims)

        y_trn -= 1  # First Class should be zero

        x_trn, x_val, y_trn, y_val = train_test_split(
            x_trn, y_trn, test_size=0.1, random_state=42
        )
        x_trn, x_tst, y_trn, y_tst = train_test_split(
            x_trn, y_trn, test_size=0.2, random_state=42
        )

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )

        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, num_cls

    elif dset_name == "census":
        np.random.seed(42)
        trn_file = os.path.join(datadir, "adult.data")
        tst_file = os.path.join(datadir, "adult.test")
        data_dims = 14
        num_cls = 2

        x_trn, y_trn = census_load(trn_file, dim=data_dims)
        x_tst, y_tst = census_load(tst_file, dim=data_dims)

        x_trn, x_val, y_trn, y_val = train_test_split(
            x_trn, y_trn, test_size=0.1, random_state=42
        )
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == "classimb":
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(
                x_trn,
                y_trn,
                x_val,
                y_val,
                x_tst,
                y_tst,
                num_cls,
                kwargs["classimb_ratio"],
            )

        elif feature == "noise":
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, num_cls

    elif dset_name == "mnist":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        mnist_transform = transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        mnist_tst_transform = transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        num_cls = 10

        fullset = torchvision.datasets.MNIST(
            root=datadir, train=True, download=True, transform=mnist_transform
        )
        testset = torchvision.datasets.MNIST(
            root=datadir, train=False, download=True, transform=mnist_tst_transform
        )

        if feature == "classimb":
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(
                np.arange(num_cls),
                size=int(kwargs["classimb_ratio"] * num_cls),
                replace=False,
            )
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(
                                torch.where(fullset.targets == i)[0].cpu().numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        subset_idxs = list(
                            torch.where(fullset.targets == i)[0].cpu().numpy()
                        )
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(
                                torch.where(fullset.targets == i)[0].cpu().numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        batch_subset_idxs = list(
                            torch.where(fullset.targets == i)[0].cpu().numpy()
                        )
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val])

        return trainset, valset, testset, num_cls

    elif dset_name == "fashion-mnist":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        mnist_transform = transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        mnist_tst_transform = transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        num_cls = 10

        fullset = torchvision.datasets.FashionMNIST(
            root=datadir, train=True, download=True, transform=mnist_transform
        )
        testset = torchvision.datasets.FashionMNIST(
            root=datadir, train=False, download=True, transform=mnist_tst_transform
        )

        if feature == "classimb":
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(
                np.arange(num_cls),
                size=int(kwargs["classimb_ratio"] * num_cls),
                replace=False,
            )
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(
                                torch.where(fullset.targets == i)[0].cpu().numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        subset_idxs = list(
                            torch.where(fullset.targets == i)[0].cpu().numpy()
                        )
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(
                                torch.where(fullset.targets == i)[0].cpu().numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        batch_subset_idxs = list(
                            torch.where(fullset.targets == i)[0].cpu().numpy()
                        )
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val])

        return trainset, valset, testset, num_cls

    elif dset_name == "cub200":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)

        if "pre_trained" in kwargs and kwargs["pre_trained"]:
            # Normalization based on imageNet
            normalize = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )
        else:
            # # Note: Normalization is calculated by using 32x32 images of the whole train set
            normalize = transforms.Normalize(
                (0.4857, 0.4996, 0.4325), (0.2067, 0.2019, 0.2422)
            )

        cub200_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        cub200_tst_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        num_cls = 200
        trainset = CUB200(root=datadir, train=True, transform=cub200_transform)
        testset = CUB200(root=datadir, train=False, transform=cub200_tst_transform)

        validation_set_fraction = 0.1
        # do stratisfied sampling to get the validation set
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_set_fraction)
        for train_index, val_index in sss.split(trainset.data, trainset.targets):
            trainset, valset = Subset(trainset, train_index), Subset(trainset, val_index)
            break
            
        return trainset, valset, testset, num_cls


    elif dset_name == "papilion":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)

        if "pre_trained" in kwargs and kwargs["pre_trained"]:
            # Normalization based on imageNet
            normalize = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )
        else:
            # # Note: Normalization is calculated by using 224x224 images of the whole train set
            normalize = transforms.Normalize(
                (0.6209, 0.6052, 0.5562), (0.2183, 0.2803, 0.3155)
            )

        pap_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        pap_tst_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        num_cls = 112
        trainset = PapilionDataset(root=datadir, train=True, transform=pap_transform)
        testset = PapilionDataset(root=datadir, train=False, transform=pap_tst_transform)

        # create a validation set with 10% of the training set.
        # Use stratisfied shuffle sampling to make sure that the validation set is balanced
        # with respect to the classes.
        validation_set_fraction = 0.1
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_set_fraction)
        for train_index, val_index in sss.split(trainset.data, trainset.targets):
            trainset, valset = Subset(trainset, train_index), Subset(trainset, val_index)
            break

        return trainset, valset, testset, num_cls

    elif dset_name == "cifar10":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        # img_size is an optional param in kwargs
        t = []
        if (
            "img_size" in kwargs
            and kwargs["img_size"] != DotMap()
            and kwargs["img_size"] is not None
        ):
            img_size = kwargs["img_size"]
        else:
            img_size = 32

        if img_size == 32:
            t += [transforms.RandomCrop(size=32)]
        else:
            t += [transforms.Resize(img_size + 32)]
            t += [transforms.RandomCrop(img_size)]
        t += [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if "pre_trained" in kwargs and kwargs["pre_trained"]:
            # Normalization based on imageNet
            normalize = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )
        else:
            # # Note: Normalization is calculated by using 32x32 images of the whole train set
            normalize = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )

        t += [normalize]
        cifar_transform = transforms.Compose(t)
        cifar_test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        num_cls = 10

        fullset = torchvision.datasets.CIFAR10(
            root=datadir, train=True, download=True, transform=cifar_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=datadir, train=False, download=True, transform=cifar_test_transform
        )

        # fullset = torch.utils.data.Subset(fullset, list(range(int(len(fullset) * 0.1))))
        # testset = torch.utils.data.Subset(testset, list(range(int(len(testset) * 0.1))))

        if feature == "classimb":
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(
                    torch.where(torch.Tensor(fullset.targets) == i)[0]
                )
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(
                np.arange(num_cls),
                size=int(kwargs["classimb_ratio"] * num_cls),
                replace=False,
            )
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(
                                torch.where(torch.Tensor(fullset.targets) == i)[0]
                                .cpu()
                                .numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        subset_idxs = list(
                            torch.where(torch.Tensor(fullset.targets) == i)[0]
                            .cpu()
                            .numpy()
                        )
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(
                                torch.where(torch.Tensor(fullset.targets) == i)[0]
                                .cpu()
                                .numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        batch_subset_idxs = list(
                            torch.where(torch.Tensor(fullset.targets) == i)[0]
                            .cpu()
                            .numpy()
                        )
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_set_fraction)
        for train_index, val_index in sss.split(fullset.data, fullset.targets):
            trainset, valset = Subset(fullset, train_index), Subset(fullset, val_index)
            break

        # num_fulltrn = len(fullset)
        # num_val = int(num_fulltrn * validation_set_fraction)
        # num_trn = num_fulltrn - num_val
        # trainset, valset = random_split(fullset, [num_trn, num_val])

        return trainset, valset, testset, num_cls

    elif dset_name == "cifar100":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        cifar100_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )

        cifar100_tst_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )

        num_cls = 100

        fullset = torchvision.datasets.CIFAR100(
            root=datadir, train=True, download=True, transform=cifar100_transform
        )
        testset = torchvision.datasets.CIFAR100(
            root=datadir, train=False, download=True, transform=cifar100_tst_transform
        )

        if feature == "classimb":
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(
                    torch.where(torch.Tensor(fullset.targets) == i)[0]
                )
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(
                np.arange(num_cls),
                size=int(kwargs["classimb_ratio"] * num_cls),
                replace=False,
            )
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(
                                torch.where(torch.Tensor(fullset.targets) == i)[0]
                                .cpu()
                                .numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        subset_idxs = list(
                            torch.where(torch.Tensor(fullset.targets) == i)[0]
                            .cpu()
                            .numpy()
                        )
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(
                                torch.where(torch.Tensor(fullset.targets) == i)[0]
                                .cpu()
                                .numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        batch_subset_idxs = list(
                            torch.where(torch.Tensor(fullset.targets) == i)[0]
                            .cpu()
                            .numpy()
                        )
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val])

        return trainset, valset, testset, num_cls

    elif dset_name == "svhn":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        svhn_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        svhn_tst_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        num_cls = 10

        fullset = torchvision.datasets.SVHN(
            root=datadir, split="train", download=True, transform=svhn_transform
        )
        testset = torchvision.datasets.SVHN(
            root=datadir, split="test", download=True, transform=svhn_tst_transform
        )

        if feature == "classimb":
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(
                    torch.where(torch.Tensor(fullset.targets) == i)[0]
                )
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(
                np.arange(num_cls),
                size=int(kwargs["classimb_ratio"] * num_cls),
                replace=False,
            )
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(
                                torch.where(torch.Tensor(fullset.targets) == i)[0]
                                .cpu()
                                .numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        subset_idxs = list(
                            torch.where(torch.Tensor(fullset.targets) == i)[0]
                            .cpu()
                            .numpy()
                        )
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(
                                torch.where(torch.Tensor(fullset.targets) == i)[0]
                                .cpu()
                                .numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        batch_subset_idxs = list(
                            torch.where(torch.Tensor(fullset.targets) == i)[0]
                            .cpu()
                            .numpy()
                        )
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val])

        return trainset, valset, testset, num_cls

    elif dset_name == "kmnist":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        kmnist_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.1904]), np.array([0.3475])),
            ]
        )

        kmnist_tst_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.1904]), np.array([0.3475])),
            ]
        )

        num_cls = 10

        fullset = torchvision.datasets.KMNIST(
            root=datadir, train=True, download=True, transform=kmnist_transform
        )
        testset = torchvision.datasets.KMNIST(
            root=datadir, train=False, download=True, transform=kmnist_tst_transform
        )

        if feature == "classimb":
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(
                    torch.where(torch.Tensor(fullset.targets) == i)[0]
                )
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(
                np.arange(num_cls),
                size=int(kwargs["classimb_ratio"] * num_cls),
                replace=False,
            )
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(
                                torch.where(torch.Tensor(fullset.targets) == i)[0]
                                .cpu()
                                .numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        subset_idxs = list(
                            torch.where(torch.Tensor(fullset.targets) == i)[0]
                            .cpu()
                            .numpy()
                        )
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(
                                torch.where(torch.Tensor(fullset.targets) == i)[0]
                                .cpu()
                                .numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        batch_subset_idxs = list(
                            torch.where(torch.Tensor(fullset.targets) == i)[0]
                            .cpu()
                            .numpy()
                        )
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val])

        return trainset, valset, testset, num_cls

    elif dset_name == "stl10":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        stl10_transform = transforms.Compose(
            [
                transforms.Pad(12),
                transforms.RandomCrop(96),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        stl10_tst_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        num_cls = 10

        fullset = torchvision.datasets.STL10(
            root=datadir, split="train", download=True, transform=stl10_transform
        )
        testset = torchvision.datasets.STL10(
            root=datadir, split="test", download=True, transform=stl10_tst_transform
        )

        if feature == "classimb":
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(
                    torch.where(torch.Tensor(fullset.targets) == i)[0]
                )
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(
                np.arange(num_cls),
                size=int(kwargs["classimb_ratio"] * num_cls),
                replace=False,
            )
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(
                                torch.where(torch.Tensor(fullset.targets) == i)[0]
                                .cpu()
                                .numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        subset_idxs = list(
                            torch.where(torch.Tensor(fullset.targets) == i)[0]
                            .cpu()
                            .numpy()
                        )
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(
                                torch.where(torch.Tensor(fullset.targets) == i)[0]
                                .cpu()
                                .numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        batch_subset_idxs = list(
                            torch.where(torch.Tensor(fullset.targets) == i)[0]
                            .cpu()
                            .numpy()
                        )
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val])

        return trainset, valset, testset, num_cls

    elif dset_name == "emnist":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        emnist_transform = transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        emnist_tst_transform = transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        num_cls = 10

        fullset = torchvision.datasets.EMNIST(
            root=datadir,
            split="digits",
            train=True,
            download=True,
            transform=emnist_transform,
        )
        testset = torchvision.datasets.EMNIST(
            root=datadir,
            split="digits",
            train=False,
            download=True,
            transform=emnist_tst_transform,
        )

        if feature == "classimb":
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(
                np.arange(num_cls),
                size=int(kwargs["classimb_ratio"] * num_cls),
                replace=False,
            )
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(
                                torch.where(fullset.targets == i)[0].cpu().numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        subset_idxs = list(
                            torch.where(fullset.targets == i)[0].cpu().numpy()
                        )
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(
                                torch.where(fullset.targets == i)[0].cpu().numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        batch_subset_idxs = list(
                            torch.where(fullset.targets == i)[0].cpu().numpy()
                        )
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)

        # validation dataset is (0.1 * train dataset)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val])
        return trainset, valset, testset, num_cls

    elif dset_name == "celeba":
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
        crop_size = 108
        re_size = 64
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[
            :,
            offset_height : offset_height + crop_size,
            offset_width : offset_width + crop_size,
        ]

        celeba_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(crop),
                transforms.ToPILImage(),
                transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )

        num_cls = 10177

        trainset = torchvision.datasets.CelebA(
            root=datadir,
            split="train",
            target_type=["identity"],
            transform=celeba_transform,
            download=True,
        )

        testset = torchvision.datasets.CelebA(
            root=datadir,
            split="test",
            target_type=["identity"],
            transform=celeba_transform,
            download=True,
        )

        valset = torchvision.datasets.CelebA(
            root=datadir,
            split="valid",
            target_type=["identity"],
            transform=celeba_transform,
            download=True,
        )

        trainset.identity.sub_(1)
        valset.identity.sub_(1)
        testset.identity.sub_(1)

        if feature == "classimb":
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(trainset.identity == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(
                np.arange(num_cls),
                size=int(kwargs["classimb_ratio"] * num_cls),
                replace=False,
            )
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(
                                torch.where(trainset.identity == i)[0].cpu().numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        subset_idxs = list(
                            torch.where(trainset.identity == i)[0].cpu().numpy()
                        )
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(
                                torch.where(trainset.identity == i)[0].cpu().numpy(),
                                size=min_samples,
                                replace=False,
                            )
                        )
                    else:
                        batch_subset_idxs = list(
                            torch.where(trainset.identity == i)[0].cpu().numpy()
                        )
                    subset_idxs.extend(batch_subset_idxs)
            trainset = torch.utils.data.Subset(trainset, subset_idxs)
        return trainset, valset, testset, num_cls

    elif dset_name == "sst2" or dset_name == "sst2_facloc":
        """
        download data/SST from https://drive.google.com/file/d/14KU6RQJpP6HKKqVGm0OF3MVxtI0NlEcr/view?usp=sharing
        or get the stanford sst data and make phrase_ids.<dev/test/train>.txt files
        pass datadir arg in dataset in config appropiriately(should look like ......../SST)
        """
        num_cls = 2
        wordvec_dim = kwargs["dataset"].wordvec_dim
        weight_path = kwargs["dataset"].weight_path
        weight_full_path = weight_path + "glove.6B." + str(wordvec_dim) + "d.txt"
        wordvec = loadGloveModel(weight_full_path)
        trainset = SSTDataset(datadir, "train", num_cls, wordvec_dim, wordvec)
        testset = SSTDataset(datadir, "test", num_cls, wordvec_dim, wordvec)
        valset = SSTDataset(datadir, "dev", num_cls, wordvec_dim, wordvec)

        return trainset, valset, testset, num_cls
    # elif dset_name == "glue_sst2":
    #     num_cls = 2
    #     raw = load_dataset("glue", "sst2")

    #     wordvec_dim = kwargs['dataset'].wordvec_dim
    #     weight_path = kwargs['dataset'].weight_path
    #     weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
    #     wordvec = loadGloveModel(weight_full_path)

    #     clean_type = 0
    #     fullset = GlueDataset(raw['train'], 'sentence', 'label', clean_type, num_cls, wordvec_dim, wordvec)
    #     # testset = GlueDataset(raw['test'], 'sentence', 'label', clean_type, num_cls, wordvec_dim, wordvec) # doesn't have gold labels
    #     valset = GlueDataset(raw['validation'], 'sentence', 'label', clean_type, num_cls, wordvec_dim, wordvec)

    #     test_set_fraction = 0.05
    #     seed = 42
    #     num_fulltrn = len(fullset)
    #     num_test = int(num_fulltrn * test_set_fraction)
    #     num_trn = num_fulltrn - num_test
    #     trainset, testset = random_split(fullset, [num_trn, num_test], generator=torch.Generator().manual_seed(seed))

    #     return trainset, valset, testset, num_cls
    elif dset_name == "sst5":
        """
        download data/SST from https://drive.google.com/file/d/14KU6RQJpP6HKKqVGm0OF3MVxtI0NlEcr/view?usp=sharing
        or get the stanford sst data and make phrase_ids.<dev/test/train>.txt files
        pass datadir arg in dataset in config appropiriately(should look like ......../SST)
        """
        num_cls = 5
        wordvec_dim = kwargs["dataset"].wordvec_dim
        weight_path = kwargs["dataset"].weight_path
        weight_full_path = weight_path + "glove.6B." + str(wordvec_dim) + "d.txt"
        wordvec = loadGloveModel(weight_full_path)
        trainset = SSTDataset(datadir, "train", num_cls, wordvec_dim, wordvec)
        testset = SSTDataset(datadir, "test", num_cls, wordvec_dim, wordvec)
        valset = SSTDataset(datadir, "dev", num_cls, wordvec_dim, wordvec)

        return trainset, valset, testset, num_cls
    elif dset_name == "trec6":
        num_cls = 6

        wordvec_dim = kwargs["dataset"].wordvec_dim
        weight_path = kwargs["dataset"].weight_path
        weight_full_path = weight_path + "glove.6B." + str(wordvec_dim) + "d.txt"
        wordvec = loadGloveModel(weight_full_path)

        cls_to_num = {"DESC": 0, "ENTY": 1, "HUM": 2, "ABBR": 3, "LOC": 4, "NUM": 5}

        trainset = Trec6Dataset(
            datadir + "train.txt", cls_to_num, num_cls, wordvec_dim, wordvec
        )
        testset = Trec6Dataset(
            datadir + "test.txt", cls_to_num, num_cls, wordvec_dim, wordvec
        )
        valset = Trec6Dataset(
            datadir + "valid.txt", cls_to_num, num_cls, wordvec_dim, wordvec
        )

        return trainset, valset, testset, num_cls
    # elif dset_name == "hf_trec6": # hugging face trec6
    #     num_cls = 6
    #     raw = load_dataset("trec")

    #     wordvec_dim = kwargs['dataset'].wordvec_dim
    #     weight_path = kwargs['dataset'].weight_path
    #     weight_full_path = weight_path+'glove.6B.' + str(wordvec_dim) + 'd.txt'
    #     wordvec = loadGloveModel(weight_full_path)

    #     clean_type = 1
    #     fullset = GlueDataset(raw['train'], 'text', 'label-coarse', clean_type, num_cls, wordvec_dim, wordvec)
    #     testset = GlueDataset(raw['test'], 'text', 'label-coarse', clean_type, num_cls, wordvec_dim, wordvec)
    #     # valset = GlueDataset(raw['validation'], num_cls, wordvec_dim, wordvec)

    #     validation_set_fraction = 0.1
    #     seed = 42
    #     num_fulltrn = len(fullset)
    #     num_val = int(num_fulltrn * validation_set_fraction)
    #     num_trn = num_fulltrn - num_val
    #     trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

    #     return trainset, valset, testset, num_cls

    else:
        raise NotImplementedError


def mean_std(loader):
    """
    Can be used to calculate normalizing paramaters. E.g:
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
    cubs_train = CUB200('/home/daankrol/data', transform=transform)
    image_data_loader = DataLoader(
        cubs_train,
        # batch size is whole dataset
        batch_size=len(cubs_train),
        shuffle=False,
        num_workers=0)

    mean, std = mean_std(image_data_loader)
    print(mean, std)
    :param loader:
    :return:
    """
    images, lebels = next(iter(loader))
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    return mean, std


# main 
if __name__ == "__main__":
    train, val, test, num_cls = gen_dataset('~/Develop/Thesis/data', 'cub200', 'dss')
    print(len(train), len(val), len(test))