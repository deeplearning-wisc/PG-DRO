import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
from PIL import Image

from celeba_dataset import CustomCelebA
from cub_dataset import CUBDataset
from mnli_dataset import MultiNLIDataset
from jigsaw_dataset import JigsawDataset



class IndexedSubset(Dataset):
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]], idx
    
    def __len__(self):
        return len(self.indices)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def prepare_data(args):
    
    if args.dataset == 'celeba':
        return prepare_celeba(args)
    elif args.dataset == 'cub':
        return prepare_cub(args)
    elif args.dataset == 'cub_prev':
        args.metadata_csv = 'metadata.csv'
        return prepare_cub(args)
    elif args.dataset == 'mnli_new':
        return prepare_mnli_new(args)
    elif args.dataset == 'jigsaw':
        return prepare_jigsaw(args)
    else:
        raise NotImplementedError
        
        
        
        
def prepare_jigsaw(args):
    
    jigsaw_train = JigsawDataset(
                    root='../datasets/civilcomments_v1.0', 
                    split='train', 
                    bias_name=args.bias_name, 
                    reverse_target=args.reverse_target, 
                    pseudo_bias=args.pseudo_bias, 
                    metadata_csv_name=f"all_data_with_identities_backtrans.csv", 
                    batch_size=args.batch_size)
    jigsaw_valid = JigsawDataset(
                    root='../datasets/civilcomments_v1.0', 
                    split='val', 
                    bias_name=args.bias_name, 
                    reverse_target=args.reverse_target, 
                    pseudo_bias=None, 
                    metadata_csv_name=f"all_data_with_identities_backtrans.csv", 
                    batch_size=args.batch_size)
    jigsaw_test = JigsawDataset(
                    root='../datasets/civilcomments_v1.0', 
                    split='test', 
                    bias_name=args.bias_name, 
                    reverse_target=args.reverse_target, 
                    pseudo_bias=None, 
                    metadata_csv_name=f"all_data_with_identities_backtrans.csv", 
                    batch_size=args.batch_size)

        
    unlabeled_indices = np.where(np.arange(len(jigsaw_train))%3 != 0)[0]
    test_indices = np.where(np.arange(len(jigsaw_train))%3 == 0)[0]
        

    group = jigsaw_valid.targets*2 + jigsaw_valid.biases
        
    valid_indices = list()
    labeled_indices = list()
    for i in range(4):
        valid_indices.append(np.where(group == i)[0][:int(len(np.where(group == i)[0])*args.val_frac//2)])
        labeled_indices.append(np.where(group == i)[0][-int(len(np.where(group == i)[0])*args.val_frac//2):])
    valid_indices = np.concatenate(valid_indices)
    labeled_indices = np.concatenate(labeled_indices)
    valid_indices.sort()
    labeled_indices.sort()
    
    
    train_labeled_dataset = Subset(jigsaw_valid, labeled_indices)
    train_unlabeled_dataset = IndexedSubset(jigsaw_train, unlabeled_indices)
    valid_dataset = Subset(jigsaw_valid, valid_indices)
    test_dataset = Subset(jigsaw_train, test_indices)
    
    print(f'Number of labeled train set: {len(labeled_indices)}')
    print(f'Number of unlabeled train set: {len(unlabeled_indices)}')
    print(f'Number of valid set: {len(valid_indices)}')
    print(f'Number of test set: {len(test_indices)}')
   
    if args.sampling == 'group_weight':
        group = np.zeros(len(labeled_indices)).astype('int')
        group[np.where(jigsaw_valid.targets[labeled_indices] == 1)[0]] += 2
        group[np.where(jigsaw_valid.biases[labeled_indices] == 1)[0]] += 1
        
        group_sample_count = np.zeros(4)
        weight = np.zeros(4)
        for g in np.unique(group):
            group_sample_count[g] = len(np.where(group == g)[0])
            weight[g] = 1. / group_sample_count[g]
        # group_sample_count = np.array([len(np.where(group == g)[0]) for g in np.unique(group)])
        # weight = 1. / group_sample_count
        samples_weight = np.array([weight[g] for g in group])
        
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                          batch_size=args.batch_size,
                                          sampler=sampler,
                                          num_workers=args.num_workers, 
                                          drop_last=True)
        
    else:
        train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers, 
                                          drop_last=True)
        
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=True, 
                                        num_workers=args.num_workers, 
                                        drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=args.num_workers)
    
    return train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader
    
        

def prepare_mnli_new(args):
    
    mnli_train = MultiNLIDataset(
                    root='../datasets/multinli', 
                    split='train', 
                    reverse_target=args.reverse_target, 
                    pseudo_bias=args.pseudo_bias, 
                    metadata_csv_name=f"metadata_random.csv")
    mnli_valid = MultiNLIDataset(
                    root='../datasets/multinli', 
                    split='val', 
                    reverse_target=args.reverse_target, 
                    pseudo_bias=None, 
                    metadata_csv_name=f"metadata_random.csv")
    mnli_test = MultiNLIDataset(
                    root='../datasets/multinli', 
                    split='test', 
                    reverse_target=args.reverse_target, 
                    pseudo_bias=None, 
                    metadata_csv_name=f"metadata_random.csv")
    
    unlabeled_indices = np.where(np.arange(len(mnli_train))%3 != 0)[0]
    test_indices = np.where(np.arange(len(mnli_train))%3 == 0)[0]
    
    if args.reverse_target:
        group = mnli_valid.targets*3 + mnli_valid.biases
    else:
        group = mnli_valid.targets*2 + mnli_valid.biases
        
    
    valid_indices = list()
    labeled_indices = list()
    for i in range(6):
        valid_indices.append(np.where(group == i)[0][:int(len(np.where(group == i)[0])*args.val_frac//2)])
        labeled_indices.append(np.where(group == i)[0][-int(len(np.where(group == i)[0])*args.val_frac//2):])
    valid_indices = np.concatenate(valid_indices)
    labeled_indices = np.concatenate(labeled_indices)
    valid_indices.sort()
    labeled_indices.sort()
    
    
    train_labeled_dataset = Subset(mnli_valid, labeled_indices)
    train_unlabeled_dataset = IndexedSubset(mnli_train, unlabeled_indices)
    valid_dataset = Subset(mnli_valid, valid_indices)
    test_dataset = Subset(mnli_train, test_indices)
    
    print(f'Number of labeled train set: {len(labeled_indices)}')
    print(f'Number of unlabeled train set: {len(unlabeled_indices)}')
    print(f'Number of valid set: {len(valid_indices)}')
    print(f'Number of test set: {len(test_indices)}')
    
    
    if args.sampling == 'group_weight':
        group = np.zeros(len(labeled_indices)).astype('int')
        if args.reverse_target:
                group[np.where(mnli_valid.targets[labeled_indices] == 1)[0]] += 3
                group[np.where(mnli_valid.biases[labeled_indices] == 1)[0]] += 1
                group[np.where(mnli_valid.biases[labeled_indices] == 2)[0]] += 2
        else:
                group[np.where(mnli_valid.targets[labeled_indices] == 1)[0]] += 2
                group[np.where(mnli_valid.targets[labeled_indices] == 2)[0]] += 4
                group[np.where(mnli_valid.biases[labeled_indices] == 1)[0]] += 1
       
        
        group_sample_count = np.zeros(6)
        weight = np.zeros(6)
        for g in np.unique(group):
            group_sample_count[g] = len(np.where(group == g)[0])
            weight[g] = 1. / group_sample_count[g]
        # group_sample_count = np.array([len(np.where(group == g)[0]) for g in np.unique(group)])
        # weight = 1. / group_sample_count
        samples_weight = np.array([weight[g] for g in group])
        
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                          batch_size=args.batch_size,
                                          sampler=sampler,
                                          num_workers=args.num_workers, 
                                          drop_last=True)
        
    else:
        train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers, 
                                          drop_last=True)
        
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=True, 
                                        num_workers=args.num_workers, 
                                        drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=args.num_workers)
    
    return train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader

        
    
    
    
def prepare_cub(args):
    
    transform_train = transforms.Compose([
                        transforms.Resize((int(args.image_size * 256.0 / 224.0), 
                                           int(args.image_size * 256.0 / 224.0))),
                        transforms.RandomCrop(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_valid = transforms.Compose([
                        transforms.Resize((int(args.image_size * 256.0 / 224.0), 
                                           int(args.image_size * 256.0 / 224.0))),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    
    
    cub_train = CUBDataset(
                    root='../datasets/cub/data/waterbird_complete95_forest2water2', 
                    split='train', 
                    reverse_target=args.reverse_target, 
                    transform=transform_train, 
                    pseudo_bias=args.pseudo_bias, 
                    metadata_csv_name=args.metadata_csv)
    cub_valid = CUBDataset(
                    root='../datasets/cub/data/waterbird_complete95_forest2water2', 
                    split='val', 
                    reverse_target=args.reverse_target, 
                    transform=transform_valid, 
                    pseudo_bias=None, 
                    metadata_csv_name=args.metadata_csv)
    cub_test = CUBDataset(
                    root='../datasets/cub/data/waterbird_complete95_forest2water2', 
                    split='test', 
                    reverse_target=args.reverse_target, 
                    transform=transform_valid, 
                    pseudo_bias=None, 
                    metadata_csv_name=args.metadata_csv)
        
    unlabeled_indices = np.where(np.arange(len(cub_train))%3 != 0)[0]
    test_indices = np.where(np.arange(len(cub_train))%3 == 0)[0]
    
    group = cub_valid.targets*2 + cub_valid.biases
    
    valid_indices = list()
    labeled_indices = list()
    for i in range(4):
        valid_indices.append(np.where(group == i)[0][:int(len(np.where(group == i)[0])*args.val_frac//2)])
        labeled_indices.append(np.where(group == i)[0][-int(len(np.where(group == i)[0])*args.val_frac//2):])
    valid_indices = np.concatenate(valid_indices)
    labeled_indices = np.concatenate(labeled_indices)
    valid_indices.sort()
    labeled_indices.sort()
    
    cub_train_unlabeled = CUBDataset(
        root='../datasets/cub/data/waterbird_complete95_forest2water2', 
        split='train', 
        reverse_target=False, 
        transform=TransformTwice(transform_train), 
        pseudo_bias=None, 
        metadata_csv_name=args.metadata_csv)
    cub_train_valid = CUBDataset(
        root='../datasets/cub/data/waterbird_complete95_forest2water2', 
        split='train', 
        reverse_target=False, 
        transform=transform_valid, 
        pseudo_bias=None, 
        metadata_csv_name=args.metadata_csv)
    cub_valid_train = CUBDataset(
        root='../datasets/cub/data/waterbird_complete95_forest2water2', 
        split='val', 
        reverse_target=False, 
        transform=transform_train, 
        pseudo_bias=None, 
        metadata_csv_name=args.metadata_csv)
    
    
    train_labeled_dataset = Subset(cub_valid_train, labeled_indices)
    train_unlabeled_dataset = IndexedSubset(cub_train_unlabeled, unlabeled_indices)
    valid_dataset = Subset(cub_valid, valid_indices)
    test_dataset = Subset(cub_train_valid, test_indices)
    
    print(f'Number of labeled train set: {len(labeled_indices)}')
    print(f'Number of unlabeled train set: {len(unlabeled_indices)}')
    print(f'Number of valid set: {len(valid_indices)}')
    print(f'Number of test set: {len(test_indices)}')
        
    
    
    if args.sampling == 'group_weight':
        group = np.zeros(len(labeled_indices)).astype('int')       
        group[np.where(cub_valid_train.targets[labeled_indices] == 1)[0]] += 2
        group[np.where(cub_valid_train.biases[labeled_indices] == 1)[0]] += 1
        group_sample_count = np.zeros(4)
        weight = np.zeros(4)
        for g in np.unique(group):
            group_sample_count[g] = len(np.where(group == g)[0])
            weight[g] = 1. / group_sample_count[g]
        # group_sample_count = np.array([len(np.where(group == g)[0]) for g in np.unique(group)])
        # weight = 1. / group_sample_count
        samples_weight = np.array([weight[g] for g in group])
        
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                          batch_size=args.batch_size,
                                          sampler=sampler,
                                          num_workers=args.num_workers, 
                                          drop_last=True)
        
    elif args.sampling == 'upsample':
        group = np.zeros(len(labeled_indices)).astype('int')
        group[np.where(cub_train.targets == 1)[0]] += 2
        group[np.where(cub_train.biases == 1)[0]] += 1
        
        weight = np.ones(4)
        weight[1] = args.upsample
        weight[2] = args.upsample
        
        samples_weight = np.array([weight[g] for g in group])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                          batch_size=args.batch_size,
                                          sampler=sampler,
                                          num_workers=args.num_workers, 
                                          drop_last=True)
        
    else:
        train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers, 
                                          drop_last=True)
        
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=True, 
                                        num_workers=args.num_workers, 
                                        drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=args.num_workers)
    
    return train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader
    
    

    



def prepare_celeba(args):
    """Create and return Dataloader."""
    

    transform_train = transforms.Compose([
                        transforms.Resize(int(args.image_size * 256.0 / 224.0)),
                        transforms.RandomCrop(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_valid = transforms.Compose([
                        transforms.Resize(int(args.image_size * 256.0 / 224.0)),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # dataset = ImageFolder(image_path, transform)
    celeba_train = CustomCelebA(
                        root='../datasets/celebA',
                        split='train', 
                        target_attr=args.target_attr, 
                        bias_attr=args.bias_attr,
                        transform=transform_train, 
                        pseudo_bias=args.pseudo_bias)
    celeba_valid = CustomCelebA(
                        root='../datasets/celebA',
                        split='valid', 
                        target_attr=args.target_attr, 
                        bias_attr=args.bias_attr,
                        transform=transform_valid)
    celeba_test = CustomCelebA(
                        root='../datasets/celebA',
                        split='test', 
                        target_attr=args.target_attr, 
                        bias_attr=args.bias_attr,
                        transform=transform_valid)
    
    
    
        
    unlabeled_indices = np.where(np.arange(len(celeba_train))%3 != 0)[0]
    test_indices = np.where(np.arange(len(celeba_train))%3 == 0)[0]
    
    group = celeba_valid.targets*2 + celeba_valid.biases
    
    valid_indices = list()
    labeled_indices = list()
    for i in range(4):
        valid_indices.append(np.where(group == i)[0][:int(len(np.where(group == i)[0])*args.val_frac//2)])
        labeled_indices.append(np.where(group == i)[0][-int(len(np.where(group == i)[0])*args.val_frac//2):])
    valid_indices = np.concatenate(valid_indices)
    labeled_indices = np.concatenate(labeled_indices)
    valid_indices.sort()
    labeled_indices.sort()
    
    
    celeba_train_unlabeled = CustomCelebA(
        root='../datasets/celebA',
        split='train', 
        target_attr=args.target_attr, 
        bias_attr=args.bias_attr,
        transform=TransformTwice(transform_train), 
        pseudo_bias=None)
    celeba_train_valid = CustomCelebA(
        root='../datasets/celebA',
        split='train', 
        target_attr=args.target_attr, 
        bias_attr=args.bias_attr,
        transform=transform_valid, 
        pseudo_bias=None)
    celeba_valid_train = CustomCelebA(
        root='../datasets/celebA',
        split='valid', 
        target_attr=args.target_attr, 
        bias_attr=args.bias_attr,
        transform=transform_train, 
        pseudo_bias=None)
    
    
    train_labeled_dataset = Subset(celeba_valid_train, labeled_indices)
    train_unlabeled_dataset = IndexedSubset(celeba_train_unlabeled, unlabeled_indices)
    valid_dataset = Subset(celeba_valid, valid_indices)
    test_dataset = Subset(celeba_train_valid, test_indices)
    
    print(f'Number of labeled train set: {len(labeled_indices)}')
    print(f'Number of unlabeled train set: {len(unlabeled_indices)}')
    print(f'Number of valid set: {len(valid_indices)}')
    print(f'Number of test set: {len(test_indices)}')
        
   
        
    if args.sampling == 'group_weight':
        group = np.zeros(len(labeled_indices)).astype('int')
        
       
        group[np.where(celeba_valid_train.targets[labeled_indices] == 1)[0]] += 2
        group[np.where(celeba_valid_train.biases[labeled_indices] == 1)[0]] += 1        
        
        group_sample_count = np.zeros(4)
        weight = np.zeros(4)
        for g in np.unique(group):
            group_sample_count[g] = len(np.where(group == g)[0])
            weight[g] = 1. / group_sample_count[g]
        # group_sample_count = np.array([len(np.where(group == g)[0]) for g in np.unique(group)])
        # weight = 1. / group_sample_count
        samples_weight = np.array([weight[g] for g in group])
        
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                          batch_size=args.batch_size,
                                          sampler=sampler,
                                          num_workers=args.num_workers, 
                                          drop_last=True)
    
    else:
        train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers, 
                                          drop_last=True)
        
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=True, 
                                        num_workers=args.num_workers, 
                                        drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=args.num_workers)
    
    return train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader