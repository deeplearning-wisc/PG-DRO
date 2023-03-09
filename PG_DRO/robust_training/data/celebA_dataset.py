import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset

class CelebADataset(ConfounderDataset):
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """

    def __init__(self, root_dir, target_name, confounder_names,
                 model_type, augment_data, pseudo_label):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.model_type = model_type

        # Read in attributes
        self.attrs_df = pd.read_csv(
            os.path.join(root_dir, 'data', 'list_attr_celeba.csv'))

        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.root_dir, 'data', 'img_align_celeba')
        self.filename_array = self.attrs_df['image_id'].values
        self.attrs_df = self.attrs_df.drop(labels='image_id', axis='columns')
        self.attr_names = self.attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        # Get the y values
        target_idx = self.attr_idx(self.target_name)
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        self.confounder_array = confounder_id
        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array_true = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')
        
        # Read in train/val/test splits
        self.split_df = pd.read_csv(
            os.path.join(root_dir, 'data', 'list_eval_partition.csv'))
        self.split_array = self.split_df['partition'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
      
        
        #Pseudo Labeling
        self.group_array = list(np.load(f'{pseudo_label}.npy'))

        # Converting test samples to original hard annotations for proper evaluation. Use pseudo labels for training
        # and validation. Best model is chosen based on validation accuracy. To stop testing after every epoch, we need
        # disable the "allow_test" flag. It is set to False by default.

        for idx, val in enumerate(self.split_array):
             if  val == 2:
                 self.group_array[idx] = self.group_array_true[idx]


       
        self.train_transform = get_transform_celebA(train=True)
        self.eval_transform = get_transform_celebA(train=False)

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)


def get_transform_celebA(train):
    
    target_size  = 64
    if train :
        transform = transforms.Compose([
                            transforms.Resize(int(target_size * 256.0 / 224.0)),
                            transforms.RandomCrop(target_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else :
        transform = transforms.Compose([
                            transforms.Resize(int(target_size * 256.0 / 224.0)),
                            transforms.CenterCrop(target_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    return transform
