import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from transformers import AutoTokenizer, BertTokenizer
from data.confounder_dataset import ConfounderDataset

class JigsawDataset(ConfounderDataset):
    """
    Jigsaw dataset. We only consider the subset of examples with identity annotations.
    Labels are 1 if target_name > 0.5, and 0 otherwise.
    95% of tokens have max_length <= 220, and 99.9% have max_length <= 300
    """

    def __init__(self, root_dir,
                 target_name, confounder_names, pseudo_label,
                 augment_data=False,
                 model_type=None):
        
        
        
        metadata_csv_name="all_data_with_identities_backtrans.csv"
        batch_size=16
    
        self.dataset_name = "jigsaw"
        self.root = root_dir
        self.target_name = 'toxicity'
        self.bias_name = "identity_any"
        self.confounder_names = confounder_names

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
        self.n_confounders = 1
        self.unique_confounders = len(np.unique(self.confounder_array))
        self.n_groups = self.n_classes * self.unique_confounders
       
        print(f"Num groups : {self.n_groups}")
        self.group_array_true = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype("int")
        
        
        self.split_dict = {"train": 0, "val": 1, "test": 2}
        self.split_array = self.metadata_df["split"].values
        self.split_array = np.array([int(self.split_dict[i]) for i in self.split_array])
       
        self.text_array = self.metadata_df["comment_text"]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        #Pseudo Labeling
        self.group_array = list(np.load(f'{pseudo_label}.npy'))

        # Converting test samples to original hard annotations for proper evaluation. Use pseudo labels for training
        # and validation. Best model is chosen based on validation accuracy. To stop testing after every epoch, we need
        # disable the "allow_test" flag. It is set to False by default.

        for idx, val in enumerate(self.split_array):
             if  val == 2:
                 self.group_array[idx] = self.group_array_true[idx]

     
            

    def __len__(self):
        return len(self.y_array)
    

    def __getitem__(self, index):
        y = self.y_array[index]
        g = self.group_array[index]

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

        return x,y,g
