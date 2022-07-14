# The following was copied from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
import os
from PIL import Image
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import json
from cords.utils.data.datasets.SL.iNaturalist_splitter import iNatSplitter

def default_loader(path):
    return Image.open(path).convert('RGB')

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic

# iNaturalist dataset of 2019
class INAT(data.Dataset):
    def __init__(self, root, partition='train'):

        if partition in ['train', 'val']:
            ann_file = 'train_val_split.json'
        else:
            ann_file = 'val2019.json'

        self.is_train = (partition == 'train')
        root = os.path.abspath(root)
        self.root = os.path.join(root, 'iNaturalist2019')
        ann_location = os.path.join(self.root, ann_file)

        print('root: {}  self.root: {} ann_location: {}'.format(root, self.root, ann_location))

        # if file train_val_split.json does not exist
        if not os.path.isfile(os.path.join(self.root, 'train_val_split.json')):
            print('validation split file not found. Making one now...')
            splitter = iNatSplitter(self.root)
            print('Constructed split file: train_val_split.json')

        if partition in ['train', 'val']:
            ann_file = 'train_val_split.json'
            ann_location = os.path.join(self.root, ann_file)

        print('Loading annotations from: ' + ann_location)
        with open(ann_location, 'r') as f:
            ann_data = json.load(f)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images'] if aa['validation'] == (partition == 'val')]
        self.ids = [aa['id'] for aa in ann_data['images'] if aa['validation'] == (partition == 'val')]

        # if we dont have class labels set them to '0'
        # FIXME: shouldn't this be 'images' instead of 'annotations'??
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations'] if aa['validation'] == (partition == 'val')]
        else:
            self.classes = [0]*len(self.imgs)

        self.num_classes = len(np.unique(self.classes))

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # print out some stats
        print('\t' + str(len(self.imgs)) + ' images')
        print('\t' + str(len(set(self.classes))) + ' classes')

        self.loader = default_loader

        # augmentation params
        self.im_size = [224, 224]  # can change this to train on higher res
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)


    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]
        tax_ids = self.classes_taxonomic[species_id]

        if self.is_train:
            img = self.scale_aug(img)
            img = self.flip_aug(img)
            img = self.color_aug(img)
        else:
            img = self.center_crop(img)

        img = self.tensor_aug(img)
        img = self.norm_aug(img)

        # return img, im_id, species_id, tax_ids
        return img, species_id

    def __len__(self):
        return len(self.imgs)