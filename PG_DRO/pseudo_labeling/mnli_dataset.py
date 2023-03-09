import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset


class MultiNLIDataset(Dataset):
    """
    MultiNLI dataset.
    label_dict = {
        'contradiction': 0,
        'entailment': 1,
        'neutral': 2
    }
    # Negation words taken from https://arxiv.org/pdf/1803.02324.pdf
    negation_words = ['nobody', 'no', 'never', 'nothing']
    """
    def __init__(
        self,
        root,
        split, 
        reverse_target, 
        pseudo_bias, 
        metadata_csv_name="metadata_random.csv",
    ):
        self.root = root
        self.split = split

        self.data_dir = os.path.join(self.root, "data")
        self.glue_dir = os.path.join(self.root, "glue_data",
                                     "MNLI")
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )
        if not os.path.exists(self.glue_dir):
            raise ValueError(
                f"{self.glue_dir} does not exist yet. Please generate the dataset first."
            )

        # Read in metadata
        # type_of_split = target_name.split("_")[-1]
        type_of_split = 'random'
        self.metadata_df = pd.read_csv(os.path.join(
            self.data_dir, metadata_csv_name),
                                       index_col=0)

        # Get the y values
        # gold_label is hardcoded
        self.y_array = self.metadata_df["gold_label"].values

        self.confounder_array = self.metadata_df['sentence2_has_negation'].values

        # Extract splits
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {"train": 0, "val": 1, "test": 2}

        # Load features
        self.features_array = []
        for feature_file in [
                "cached_train_bert-base-uncased_128_mnli",
                "cached_dev_bert-base-uncased_128_mnli",
                "cached_dev_bert-base-uncased_128_mnli-mm",
        ]:

            features = torch.load(os.path.join(self.glue_dir, feature_file))
           # print(features)
            self.features_array += features

        self.all_input_ids = torch.tensor(
            [f.input_ids for f in self.features_array], dtype=torch.long)
        print(self.all_input_ids.shape)
        self.all_input_masks = torch.tensor(
            [f.input_mask for f in self.features_array], dtype=torch.long)
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in self.features_array], dtype=torch.long)
        self.all_label_ids = torch.tensor(
            [f.label_id for f in self.features_array], dtype=torch.long)

        self.x_array = torch.stack(
            (self.all_input_ids, self.all_input_masks, self.all_segment_ids),
            dim=2)
        print(self.x_array.shape)

        assert np.all(np.array(self.all_label_ids) == self.y_array)
        
        # Load augmented features
        # self.aug_features_array = []
        # for aug_feature_file in [
        #         "mnli_backtrans_bert.npy",
        #         "mnli_backtrans_bert_dev.npy",
        #         "mnli_backtrans_bert_dev_mm.npy",
        # ]:
        #     aug_features = np.load(os.path.join(self.glue_dir, aug_feature_file))

        #     self.aug_features_array.append(aug_features)
            
        # self.x_aug_array = torch.from_numpy(np.concatenate(self.aug_features_array))

        self.x_aug_array = torch.load(os.path.join(self.glue_dir, 'backtrans'))
        
        
        # split
        assert split in ("train", "val",
                         "test"), f"{split} is not a valid split"
        mask = self.split_array == self.split_dict[split]

        num_split = np.sum(mask)
        indices = np.where(mask)[0]
        
        self.x_array = self.x_array[indices]
        self.x_aug_array = self.x_aug_array[indices]
        
        if reverse_target:
            self.targets = self.confounder_array[indices]
            self.biases = self.y_array[indices]
        else:
            self.targets = self.y_array[indices]
            self.biases = self.confounder_array[indices]
            
        if pseudo_bias is not None:
            self.biases = torch.load(f'pseudo_bias/mnli/{pseudo_bias}.pth').numpy()
        

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        y = self.targets[index]
        a = self.biases[index]
        x = self.x_array[index, ...]
        x_aug = self.x_aug_array[index, ...]
        
        ret_obj = {'x': x,
                   'x_aug': x_aug, 
                   'y': y,
                   'a': a,
                   'dataset_index': index,
                   }

        return ret_obj