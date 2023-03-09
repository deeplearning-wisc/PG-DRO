import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd

class CustomCelebA():

    def __init__(self, root, split, target_attr, bias_attr, transform, pseudo_bias =None):

        self.root_dir = root
        # Read in attributes
        self.attrs_df = pd.read_csv(
            os.path.join(root, 'data', 'list_attr_celeba.csv'))

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
        target_idx = target_attr
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = bias_attr
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders 
        self.confounder_array = confounder_id
     #   print(self.confounder_array)
     #   print(self.confounder_idx)

        # Map to groups
       # self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
     #   self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')
        self.transform = transform
     #   self.group_array = pd.read_csv(os.path.join(root_dir, 'data', 'group_attrs_celeba.csv'))['group'].values
       
        # Read in train/val/test splits
        self.split_df = pd.read_csv(
            os.path.join(root,'data', 'list_eval_partition.csv'))
        self.split_array = self.split_df['partition'].values
        self.split_dict = {
            'train': 0,
            'valid': 1,
            'test': 2
        }  
        assert split in ("train", "valid",
                         "test"), f"{split} is not a valid split"
        mask = self.split_array == self.split_dict[split]

        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        self.filename = self.filename_array[indices]
        self.targets = self.y_array[indices]
        self.biases = self.confounder_array[indices]

        if pseudo_bias is not None:
            self.biases = torch.load(f'pseudo_bias_{pseudo_bias}.pth')


    def __len__(self):

        return len(self.filename)
    def __getitem__(self, index):
      
        X = Image.open(os.path.join(self.root_dir, 'data', "img_align_celeba", self.filename[index]))
        y = self.targets[index]
        a = self.biases[index]
        
        if self.transform is not None:
            X = self.transform(X)
            
        ret_obj = {'x': X,
                   'y': y,
                   'a': a,
                   'dataset_index': index,
                   'filename': self.filename[index],
                   }

        return ret_obj