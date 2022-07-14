# The following was copied from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
import os
from PIL import Image
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import json

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

# Constructing a stratisfied 10% validation split of the training dataset
class iNatSplitter():
    def __init__(self, root):
        partition = 'train'
        ann_file = partition + '2019.json'
        self.is_train = (partition == 'train')
        root = os.path.abspath(root)
        self.root = os.path.join(root, 'iNaturalist2019')
        ann_location = os.path.join(self.root, ann_file)

        print('root: {}  self.root: {} ann_location: {}'.format(root, self.root, ann_location))

        if partition == 'test':
            self.root = os.path.join(root, 'iNaturalist2019/')
        else:
            self.root = os.path.join(root, 'iNaturalist2019/')

        print('self.root: {}'.format(self.root))

        print('Loading annotations from: ' + ann_location)
        with open(ann_location, 'r') as f:
            ann_data = json.load(f)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # print out some stats
        print('\t' + str(len(self.imgs)) + ' images')
        print('\t' + str(len(set(self.classes))) + ' classes')


        # read the training json file 
        with open(ann_location, 'r') as f:
            ann_data = json.load(f)

        # train json looks like this:
        # {
        #  "info": {}
        # "images": [
        #  {
        #  "id": 265213,
        # "file_name": "image1.jpg",
        # "width": 640,
        # "height": 480,]
        # "annotations": [{ "image_id": 265213, "category_id": 644, "id": 265213 },]
        # "categories": []
        # }

        # so we can get the class labels from the annotations by using the category id of the image 
        self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        # count the number of images per class
        self.class_counts = {}
        for cc in np.unique(self.classes):
            self.class_counts[cc] = np.sum(self.classes == cc)
        
        should_be_marked_per_class = {}
        for cc in np.unique(self.classes):
            should_be_marked_per_class[cc] = int(self.class_counts[cc] * 0.1)

        print('class counts: {}'.format(self.class_counts))
        print('should be marked per class: {}'.format(should_be_marked_per_class))

        already_marked = [0 for cc in np.unique(self.classes)]

        # Now iterate all images, and mark them as validation in the json (by adding the key 'validation' to the image) if 
        # for that class we have not yet marked the required number of images.
        for ii, img in enumerate(ann_data['images']):
            class_label = self.classes[ii]
            current_count = already_marked[class_label]
            if current_count < should_be_marked_per_class[class_label]:
                img['validation'] = True
                already_marked[class_label] += 1
            else:
                img['validation'] = False

        # write the new json file with name "train_val_split.json"
        with open(os.path.join(self.root, 'train_val_split.json'), 'w') as f:
            json.dump(ann_data, f)

if __name__ == '__main__':
    splitter = iNatSplitter('/home/daankrol/data')