import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', default='temp', type=str)

# dataset configuration
parser.add_argument('--dataset', default='celeba', type=str, help='dataset celeba[default]')
parser.add_argument('--image_size', default=64, type=int)
parser.add_argument('--split', default=None, type=str)
parser.add_argument('--reverse_target', action='store_true')
parser.add_argument('--target_attr', default=9, type=int)
parser.add_argument('--bias_attr', default=20, type=int)
parser.add_argument('--pseudo_bias', default=None, type=str)
parser.add_argument('--sampling', default=None, type=str)

parser.add_argument('--num_classes', default=2, type=int)

parser.add_argument('--metadata_csv', default='metadata_new.csv', type=str)

parser.add_argument('--val_frac', default=0.2, type=float)

parser.add_argument('--bias_name', default='identity_any', type=str)


# model configuration
parser.add_argument('--model', default='resnet18', type=str)
parser.add_argument('--pretrained', default=None, type=str)
parser.add_argument('--num_branches', default=1, type=int)
parser.add_argument('--linear', default=None, type=str)


# DARP
parser.add_argument('--darp', action='store_true', help='Applying DARP')
parser.add_argument('--warmup', type=int, default=200, help='Number of warm up epoch for DARP')
parser.add_argument('--delta', default=2.0, type=float, help='hyperparameter for removing noisy entries')
parser.add_argument('--est', action='store_true', help='Using estimated distribution for unlabeled dataset')
parser.add_argument('--iter_T', type=int, default=10, help='Number of iteration (T) for DARP')
parser.add_argument('--num_iter', type=int, default=10, help='Scheduling for updating pseudo-labels')


# VAT
parser.add_argument('--vat_xi', type=float, default=10.0)
parser.add_argument('--vat_eps', type=float, default=1.0)
parser.add_argument('--vat_ip', type=int, default=1)
parser.add_argument('--vat_alpha', type=float, default=1.0)

# MixMatch
parser.add_argument('--T', type=float, default=0.5, help='Temperature (T) for MixMatch')
parser.add_argument('--mix_alpha', type=float, default=0.75, help='Alpha for Mixup')
parser.add_argument('--lambda_u', type=float, default=75, help='consistency coefficient for MixMatch')

# FixMatch
parser.add_argument('--tau', default=0.95, type=float, help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--reweight_unlabeled', action='store_true')
parser.add_argument('--pseudo_balance', action='store_true')


# optimization configuration
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', '--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--lr_decay', nargs='+', type=int)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')

parser.add_argument('--val_iteration', default=100, type=int)

parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', type=int, default=1)


def get_arguments():
    args = parser.parse_args()

    return args