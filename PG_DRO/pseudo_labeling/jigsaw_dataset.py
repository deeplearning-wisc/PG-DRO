import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from transformers import AutoTokenizer, BertTokenizer


class JigsawDataset(Dataset):
    """
    Jigsaw dataset. We only consider the subset of examples with identity annotations.
    Labels are 1 if target_name > 0.5, and 0 otherwise.
    95% of tokens have max_length <= 220, and 99.9% have max_length <= 300
    """

    def __init__(
        self,
        root,
        bias_name,
        split, 
        reverse_target, 
        pseudo_bias, 
        metadata_csv_name="all_data_with_identities.csv",
        batch_size=None,
    ):
        # def __init__(self, args):
        self.dataset_name = "jigsaw"
        # self.aux_dataset = args.aux_dataset
        self.root = root
        self.target_name = 'toxicity'
        self.bias_name = bias_name
        # self.confounder_names = confounder_names
        # self.augment_data = augment_data

        if batch_size == 32:
            self.max_length = 128
        elif batch_size == 24:
            self.max_length = 220
        elif batch_size == 16:
            self.max_length = 300
        elif batch_size == 8:
            self.max_length = 300
        else:
            assert False, "Invalid batch size"

        # assert self.augment_data == False
        # assert self.model in ["bert-base-cased", "bert-base-uncased"]
        
        self.data_dir = os.path.join(self.root, "data")
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )

        # Read in metadata
        data_filename = metadata_csv_name
        print("metadata_csv_name:", metadata_csv_name)

        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, data_filename), index_col=0
        )

        # Get the y values
        self.y_array = (self.metadata_df[self.target_name].values >= 0.5).astype("long")
        self.n_classes = len(np.unique(self.y_array))
        
        self.confounder_array = (self.metadata_df[self.bias_name].values >= 0.5).astype("long")
        
        self.eval_attrs = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white']
        self.eval_attr_arrays = {}
        for attr_name in self.eval_attrs:
            self.eval_attr_arrays[attr_name] = (self.metadata_df[attr_name].values >= 0.5).astype("long")

#         if self.confounder_names[0] == "only_label":
#             self.n_groups = self.n_classes
#             self.group_array = self.y_array
#         else:
#             # Confounders are all binary
#             # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
#             self.n_confounders = len(self.confounder_names)
#             confounders = (self.metadata_df.loc[:, self.confounder_names] >= 0.5).values
#             self.confounder_array = confounders @ np.power(
#                 2, np.arange(self.n_confounders)
#             )

#             # Map to groups
#             self.n_groups = self.n_classes * pow(2, self.n_confounders)
#             self.group_array = (
#                 self.y_array * (self.n_groups / 2) + self.confounder_array
#             ).astype("int")

        # Extract splits
        self.split_dict = {"train": 0, "val": 1, "test": 2}
        self.split_array = self.metadata_df["split"].values

        # Extract text
        # self.text_array = list(self.metadata_df["comment_text"])
        # self.text_aug_array = list(self.metadata_df['backtrans'])
        self.text_array = self.metadata_df["comment_text"]
        self.text_aug_array = self.metadata_df['backtrans']
        # self.tokenizer = BertTokenizer.from_pretrained(self.model)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # split
        assert split in ("train", "val",
                         "test"), f"{split} is not a valid split"
        mask = self.split_array == split

        num_split = np.sum(mask)
        indices = np.where(mask)[0]
        
        self.text_array = list(self.text_array[indices])
        self.text_aug_array = list(self.text_aug_array[indices])
        # TODO: add backtrans
        
        if reverse_target:
            self.targets = self.confounder_array[indices]
            self.biases = self.y_array[indices]
        else:
            self.targets = self.y_array[indices]
            self.biases = self.confounder_array[indices]
            
        for attr_name in self.eval_attrs:
            self.eval_attr_arrays[attr_name] = self.eval_attr_arrays[attr_name][indices]
            
        if pseudo_bias is not None:
            self.biases = torch.load(f'pseudo_bias/jigsaw/{pseudo_bias}.pth').numpy()
            

    def __len__(self):
        return len(self.targets)
    

    def __getitem__(self, index, return_eval=False):
        y = self.targets[index]
        a = self.biases[index]

        text = self.text_array[index]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )  # 220
        x = torch.stack(
            (tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"]),
            dim=2,
        )
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        
        text_aug = self.text_aug_array[index]
        tokens_aug = self.tokenizer(
            text_aug,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )  # 220
        x_aug = torch.stack(
            (tokens_aug["input_ids"], tokens_aug["attention_mask"], tokens_aug["token_type_ids"]),
            dim=2,
        )
        x_aug = torch.squeeze(x_aug, dim=0)  # First shape dim is always 1

        ret_obj = {'x': x,
                   'x_aug': x_aug, 
                   'y': y,
                   'a': a,
                   'dataset_index': index,
                   }
        
        if return_eval:
            for attr_name in self.eval_attrs:
                ret_obj[attr_name] = self.eval_attr_arrays[attr_name][index]

        return ret_obj
