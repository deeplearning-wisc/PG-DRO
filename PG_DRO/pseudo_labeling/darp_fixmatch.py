# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
# import sklearn.metrics as sm
# import pandas as pd
# import sklearn.metrics as sm
import random
import numpy as np

from pytorch_transformers import AdamW, WarmupLinearSchedule

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from resnet import get_model
from data_loader import prepare_data
from arguments import get_arguments

torch.multiprocessing.set_sharing_strategy('file_system')

args = get_arguments()
use_cuda = True
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')

print()
print(args)

def build_model():
    
    model = get_model(args)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if args.lr_decay is not None:
        for decay_epoch in args.lr_decay:
            lr *= (0.1 ** int(epoch >= decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def estimate_pseudo(M_k, y_m):
    pseudo_labels = torch.zeros_like(y_m)
    k_probs = torch.zeros(args.num_classes)

    for k in np.argsort(M_k):
        delta_hat = int(args.delta * M_k[k])
        sorted_probs, idx = y_m[:, k].sort(dim=0, descending=True)
        pseudo_labels[idx[:delta_hat], k] = 1
        k_probs[k] = sorted_probs[:delta_hat].sum()

    return pseudo_labels, (M_k + 1e-6) / (k_probs + 1e-6)


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

    
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)

        return Lx, Lu



def test(model, test_loader, writer, epoch):
    model.eval()
    correct = 0
    test_loss = 0
    
    ys = []
    bs = []
    test_losses = []
    corrects = []
    corrects_bias = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs, targets = batch['x'].to(device), batch['a'].to(device)
            if args.model.startswith('bert'):
                input_ids = inputs[:, :, 0]
                input_masks = inputs[:, :, 1]
                segment_ids = inputs[:, :, 2]
                y_hat = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=targets,
                )[1]  # [1] returns logits
            else:
                # outputs.shape: (batch_size, num_classes)
                y_hat = model(inputs)
            test_loss = F.cross_entropy(y_hat, targets, reduction='none')
            _, predicted = y_hat.cpu().max(1)
            if args.num_classes == 2:
                correct = predicted.eq(batch['y'])
                correct_bias = predicted.eq(batch['a'])
            elif args.num_classes == 4:
                correct = (predicted//2).eq(batch['y'])
                correct_bias = (predicted%2).eq(batch['a'])
            
            test_losses.append(test_loss.cpu())
            corrects.append(correct)
            corrects_bias.append(correct_bias)
            ys.append(batch['y'])
            bs.append(batch['a'])
            
    test_losses = torch.cat(test_losses)
    corrects = torch.cat(corrects)
    corrects_bias = torch.cat(corrects_bias)
    ys = torch.cat(ys)
    bs = torch.cat(bs)
    
    group = ys*2 + bs
    group_indices = dict()
    
    num_classes = test_loader.dataset.dataset.targets.max().item() + 1
    num_biases = test_loader.dataset.dataset.biases.max().item() + 1
    
    for i in range(num_classes*num_biases):
        group_indices[i] = np.where(group == i)[0]
        
    
    print('')
    worst_accuracy = 101
    worst_bias_accuracy = 101
    for i in range(num_classes*num_biases):
        loss = test_losses[group_indices[i]].mean().item()
        correct = corrects[group_indices[i]].sum().item()
        correct_bias = corrects_bias[group_indices[i]].sum().item()
        accuracy = 100. * correct / len(group_indices[i])
        accuracy_bias = 100. * correct_bias / len(group_indices[i])
        if accuracy < worst_accuracy:
            worst_accuracy = accuracy
            worst_accuracy_bias = accuracy_bias
            worst_loss = loss
            worst_correct = correct
            worst_correct_bias = correct_bias
            worst_len = len(group_indices[i])
        if accuracy_bias < worst_bias_accuracy:
            worst_bias_accuracy = accuracy_bias
            worst_bias_correct = correct_bias
            worst_bias_len = len(group_indices[i])
        
        writer.add_scalar(f'valid/accuracy_group{i}', accuracy, epoch)
        writer.add_scalar(f'valid/accuracy_bias_group{i}', accuracy_bias, epoch)
        print(f'Test set - group {i}: Average loss: {loss:.4f}, Accuracy: {correct}/{len(group_indices[i])}({accuracy:.4f}%)')
        print(f'Test set - group {i}: Bias Accuracy: {correct_bias}/{len(group_indices[i])}({accuracy_bias:.4f}%)\n')
        
    writer.add_scalar(f'valid/accuracy_worst_group', worst_accuracy, epoch)
    writer.add_scalar(f'valid/bias_accuracy_worst_group', worst_bias_accuracy, epoch)
    print(f'Test set - worst group: Average loss: {worst_loss:.4f}, Accuracy: {worst_correct}/{worst_len}({worst_accuracy:.4f}%)\n')
    print(f'Test set - worst group: Bias Accuracy: {worst_correct_bias}/{worst_len}({worst_accuracy_bias:.4f}%)\n')
    print(f'Test set - worst bias group: Bias Accuracy: {worst_bias_correct}/{worst_bias_len}({worst_bias_accuracy:.4f}%)\n')
    
    loss = test_losses.mean().item()
    correct = corrects.sum().item()
    correct_bias = corrects_bias.sum().item()
    accuracy = 100. * corrects.sum().item() / len(test_loader.dataset)
    accuracy_bias = 100. * corrects_bias.sum().item() / len(test_loader.dataset)
    writer.add_scalar(f'valid/accuracy_average', accuracy, epoch)
    writer.add_scalar(f'valid/accuracy_bias_average', accuracy_bias, epoch)
    print(f'Test set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.4f}%)\n')
    print(f'Test set: Bias Accuracy: {correct_bias}/{len(test_loader.dataset)} ({accuracy_bias:.4f}%)\n')

    return worst_bias_accuracy




def train(labeled_loader, unlabeled_loader, model, optimizer_model, epoch, pseudo_orig, pseudo_refine):
    print('\nEpoch: %d' % epoch)
    
    losses = 0
    losses_x = 0
    losses_u = 0
    
    criterion = SemiLoss()
    # criterion = nn.CrossEntropyLoss()

    labeled_loader_iter = iter(labeled_loader)
    unlabeled_loader_iter = iter(unlabeled_loader)
    
    num_classes = unlabeled_loader.dataset.dataset.targets.max().item() + 1
    num_biases = unlabeled_loader.dataset.dataset.biases.max().item() + 1
    
    labeled_indices = labeled_loader.dataset.indices
    labeled_targets = labeled_loader.dataset.dataset.targets[labeled_indices]
    labeled_biases = labeled_loader.dataset.dataset.biases[labeled_indices]
    labeled_groups = labeled_targets*2 + labeled_biases
        
    labeled_group_pop = torch.Tensor([(labeled_groups == i).sum() for i in range(num_classes*num_biases)])
    
    print(labeled_group_pop)
    min_group = labeled_group_pop.argmin().item()
    group_tau = torch.ones(num_classes*num_biases)
    # group_tau = np.ones(num_classes*num_biases)
    group_tau[min_group] = args.tau
    
    unlabeled_indices = unlabeled_loader.dataset.indices
    unlabeled_targets = unlabeled_loader.dataset.dataset.targets[unlabeled_indices]
        
    group_weight = torch.ones((num_classes, num_biases))
    
    pseudo_count = np.zeros(num_classes*num_biases)
    pseudo_count_balanced = np.zeros(num_classes*num_biases)
    
    for batch_idx in range(args.val_iteration):
        model.train()
        
        # load data
        try:
            batch = next(labeled_loader_iter)
        except:
            labeled_loader_iter = iter(labeled_loader)
            batch = next(labeled_loader_iter)
        inputs_x = batch['x'].to(device)
        batch_size = inputs_x.size(0)
        
        
        # targets_x = (batch['y']*2 + batch['a'])
        if args.num_classes == 2:
            targets_x = batch['a']
        elif args.num_classes == 4:
            targets_x = (batch['y']*2 + batch['a'])
        targets_x = torch.zeros(batch_size, args.num_classes).scatter_(1, targets_x.view(-1, 1), 1).to(device)
        
        # index = batch['dataset_index']
        
        try:
            batch, idx_u = next(unlabeled_loader_iter)
            if args.model.startswith('bert'):
                inputs_u = batch['x']
                inputs_u2 = batch['x_aug']
            else:
                inputs_u, inputs_u2 = batch['x']
        except StopIteration:
            unlabeled_loader_iter = iter(unlabeled_loader)
            batch, idx_u = next(unlabeled_loader_iter)
            if args.model.startswith('bert'):
                inputs_u = batch['x']
                inputs_u2 = batch['x_aug']
            else:
                inputs_u, inputs_u2 = batch['x']
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)
                
        
        # generate pseudo labels by aggregation and sharpening
        with torch.no_grad():
            
            if args.model.startswith('bert'):
                input_ids = inputs_u[:, :, 0]
                input_masks = inputs_u[:, :, 1]
                segment_ids = inputs_u[:, :, 2]
                outputs_u = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                )[0]  # [1] returns logits
                                
                input_ids = inputs_u2[:, :, 0]
                input_masks = inputs_u2[:, :, 1]
                segment_ids = inputs_u2[:, :, 2]
                outputs_u2 = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                )[0]  # [1] returns logits
                
                targets_u = torch.softmax(outputs_u, dim=1)
            else:
                # outputs.shape: (batch_size, num_classes)
                outputs_u = model(inputs_u)
                outputs_u2 = model(inputs_u2)
                targets_u = torch.softmax(outputs_u, dim=1)
            
            
            # update the saved predictions with current one
            p = targets_u
            pseudo_orig[idx_u, :] = p.data.cpu()
            pseudo_orig_backup = pseudo_orig.clone()
            
            # applying DARP
            if args.darp and epoch > args.warmup:
                # iterative normalization
                for i in range(30):
                    selected_u, weights_u = estimate_pseudo(target_dist, pseudo_orig)
                    scale_term = selected_u * weights_u.reshape(1, -1)
                    pseudo_orig = (pseudo_orig * scale_term + 1e-6) / (pseudo_orig * scale_term + 1e-6).sum(dim=1, keepdim=True)
                    
                targets_u = pseudo_orig[idx_u].to(device)
                pseudo_orig = pseudo_orig_backup
                
        
        max_p, p_hat = torch.max(targets_u, dim=1)
        if args.pseudo_balance:
            a_hat = p_hat.cpu()
        p_hat = torch.zeros(batch_size, args.num_classes).cuda().scatter_(1, p_hat.view(-1, 1), 1)

        select_mask = max_p.ge(args.tau).float()
        
        selected_indices = np.where(select_mask.cpu())[0]
        group_selected = (batch['y']*2 + batch['a'])[selected_indices]
        for i in range(num_classes*num_biases):
            pseudo_count[i] += (group_selected == i).sum()
            
            
        if args.pseudo_balance:
            
            select_mask_balanced = torch.zeros_like(select_mask)
            group_pseudo = batch['y']*2 + a_hat
            
            i = min_group
            c = min_group // num_biases
            b = min_group % num_biases
            target_indices = np.where(unlabeled_targets == c)[0]
            unlabeled_min_pop = (pseudo_orig[target_indices, b] >= args.tau).sum().item()
            
            total_min_pop = labeled_group_pop[min_group] + unlabeled_min_pop
            
            for i in range(num_classes*num_biases):

                c = i // num_biases
                b = i % num_biases
                
                # compute group threshold
                if i != min_group:
                    target_indices = np.where(unlabeled_targets == c)[0]
                    v, _ = pseudo_orig[target_indices, b].sort(descending=True)
                    topk = int(total_min_pop - labeled_group_pop[i])
                    if topk > 0:
                        group_tau[i] = v[topk]
                
                # thresholding per group
                selected_indices = np.where(group_pseudo == i)[0]
                select_mask_balanced[selected_indices] += targets_u[selected_indices, b].ge(group_tau[i]).float()
                # select_mask_balanced[selected_indices] += (targets_u[selected_indices, b] > group_tau[i]).float()
                
            select_mask = select_mask_balanced
            
            selected_indices = np.where(select_mask.cpu())[0]
            group_selected = (batch['y']*2 + batch['a'])[selected_indices]
            for i in range(num_classes*num_biases):
                pseudo_count_balanced[i] += (group_selected == i).sum()
            
        
        upweight_mask = torch.ones_like(max_p)
            
            
        if args.model.startswith('bert'):
            all_input_ids = torch.cat([inputs_x[:, :, 0], inputs_u2[:, :, 0]])
            all_input_masks = torch.cat([inputs_x[:, :, 1], inputs_u2[:, :, 1]])
            all_segment_ids = torch.cat([inputs_x[:, :, 2], inputs_u2[:, :, 2]])
            all_targets = torch.cat([targets_x, p_hat], dim=0)
            
            all_outputs = model(
                input_ids=all_input_ids,
                attention_mask=all_input_masks,
                token_type_ids=all_segment_ids,
            )[0]  # [1] returns logits
            logits_x = all_outputs[:batch_size]
            logits_u = all_outputs[batch_size:]
        else:
            all_inputs = torch.cat([inputs_x, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, p_hat], dim=0)

            all_outputs = model(all_inputs)
            logits_x = all_outputs[:batch_size]
            logits_u = all_outputs[batch_size:]

        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask*upweight_mask)
        loss = Lx + Lu

        
        # record loss
        
        # compute gradient and do SGD step
        if args.model.startswith("bert"):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()
            optimizer.step()
            model.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses += loss.item()
        losses_x += Lx.item()
        losses_u += Lu.item()

        if (batch_idx + 1) % 5 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss_x: %.4f\t'
                  'Loss_u: %.4f\t' % (
                      (epoch + 1), args.epochs, batch_idx + 1, args.val_iteration, 
                      (losses_x / (batch_idx + 1)), (losses_u / (batch_idx + 1))))
            print(pseudo_count)
            # print(group_weight)
            if args.pseudo_balance:
                print(pseudo_count_balanced)
                print(group_tau)
                
    return losses/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1), pseudo_orig, pseudo_refine, pseudo_count, pseudo_count_balanced


train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader = prepare_data(args)
if args.dataset == 'celeba':
    if args.split == 'semi03':
        target_dist = np.array([47750, 44617, 15240, 906])
elif args.dataset == 'cub':
    if args.split == 'semi03':
        target_dist = np.array([1060, 171, 42, 326])
elif args.dataset == 'cub_prev':
    if args.split == 'semi03':
        target_dist = np.array([1065, 64, 20, 350])
# args.num_classes = 4




# create model
model = build_model()

if args.model == 'bert':
    
    args.max_grad_norm = 1.0
    args.adam_epsilon = 1e-8
    args.warmup_steps = 0
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr,
                      eps=args.adam_epsilon)
    t_total = len(train_labeled_loader) * args.epochs
    print(f"\nt_total is {t_total}\n")
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=args.warmup_steps,
                                     t_total=t_total)

else:

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    else:
        raise NotImplementedError
        
    scheduler = None


ckpt_dir = os.path.join('results', args.dataset, args.name)
log_dir = os.path.join('summary', args.dataset, args.name)
    
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
writer = SummaryWriter(log_dir)


def main():
    best_acc = 0
    pseudo_orig = torch.ones(len(train_unlabeled_loader.dataset), args.num_classes) / args.num_classes
    pseudo_refine = torch.ones(len(train_unlabeled_loader.dataset), args.num_classes) / args.num_classes
    
    for epoch in range(args.epochs):
        
        if args.darp and (epoch == args.warmup):
            best_acc = 0
        
        adjust_learning_rate(optimizer, epoch)
        train_loss, train_loss_x, train_loss_u, pseudo_orig, pseudo_refine, pseudo_count, pseudo_count_balanced = train(train_labeled_loader, 
                                                                                                                        train_unlabeled_loader, 
                                                                                                                        model, optimizer, epoch, 
                                                                                                                        pseudo_orig, pseudo_refine)
        writer.add_scalar(f'train/train_loss', train_loss, epoch)
        writer.add_scalar(f'train/train_loss_labeled', train_loss_x, epoch)
        writer.add_scalar(f'train/train_loss_unlabeled', train_loss_u, epoch)
        
        for i in range(len(pseudo_count)):
            writer.add_scalar(f'train/pseudo_count_group{i}', pseudo_count[i], epoch)
            
        if args.pseudo_balance:
            for i in range(len(pseudo_count_balanced)):
                writer.add_scalar(f'train/pseudo_count_balanced_group{i}', pseudo_count_balanced[i], epoch)
                
        valid_acc = test(model, valid_loader, writer, epoch)
        
        if valid_acc >= best_acc:
            best_acc = valid_acc
            state_dict = {'model': model.state_dict()}
            torch.save(state_dict, os.path.join(ckpt_dir, f'best_model.pth'))

    print('best accuracy:', best_acc)


if __name__ == '__main__':
    main()
