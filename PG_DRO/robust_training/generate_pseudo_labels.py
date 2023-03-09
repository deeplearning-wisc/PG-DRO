import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np

group_dict = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3,
            (2, 0): 4,
            (2, 1): 5,
        }

def find_prob(y,pred_bias,num_groups):
    probs = np.zeros(num_groups)
    probs[group_dict[(y,1)]] = pred_bias[1] 
    probs[group_dict[(y,0)]] = pred_bias[0] 
    return probs

def generate_group_prob(model, dataloader, dataset, save_name):
    group_prob = []
    num_groups = 6 if dataset == "multinli" else 4
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
           
            inputs, targets = batch

            if dataset == "multinli" or dataset == "civilcomments":
                input_ids = inputs[:, :, 0]
                input_masks = inputs[:, :, 1]
                segment_ids = inputs[:, :, 2]
            
                y_hat = F.softmax(model(
                        input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids,
                    )[0])

            elif dataset == "waterbirds" or dataset == "celebA":

                y_hat = F.softmax(model(inputs.cuda()))

            prob = [find_prob(targets[i].cpu().item(),y_hat[i].cpu().numpy(), num_groups) for i in range(len(y_hat))]
            group_prob.extend(prob)
    
    with open(f'{save_name}.npy', 'wb') as f:
        np.save(f, group_prob)
        
    return group_prob
            