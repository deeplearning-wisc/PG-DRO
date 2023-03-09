# PG-DRO: Distributionally Robust Optimization with Probabilistic Group
This codebase provides a Pytorch implementation for the paper: PG-DRO: Distributionally Robust Optimization with Probabilistic Group. 

## Abstract
Modern machine learning models may be susceptible to learning spurious correlations that hold on average but not for the atypical group of samples. To address the problem, previous approaches minimize the empirical worst-group risk. Despite the promise, they often assume that each sample belongs to one and only one group, which can be restrictive in real-world scenarios. In this paper, we propose a novel framework PG-DRO, which explores the idea of probabilistic group membership for distributionally robust optimization. Key to our framework, we consider soft group membership instead of hard group annotations. The group probabilities can be flexibly generated using either semi-supervised learning or zero-shot approaches. Our framework accommodates samples with group membership ambiguity, offering stronger flexibility and generality than the prior art. We comprehensively evaluate PG-DRO on both natural language processing and image classification benchmarks, establishing superior performance. With minimal group supervision, PG-DRO outperforms G-DRO (with fully annotated group labels) by 10.4% on the CivilComments-WILDS benchmark.

The experiments use the following datasets:

- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Waterbirds
- [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)
- [CivilComments-WILDS](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)

## Required Packages
Our experiments are conducted on Ubuntu Linux 20.04 with Python 3.7.4 and Pytorch 1.9.0. Besides, the following packages are required to be installed:

* Scipy
* Numpy
* Sklearn
* Pandas
* tqdm
* pillow
* timm
* pytorch_transformers
* torchvision

## Datasets and Code

### WaterBirds
Similar to the construction in [Group_DRO](https://github.com/kohpangwei/group_DRO), this dataset is constructed by cropping out birds from photos in the Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011) and transferring them onto backgrounds from the Places dataset (Zhou et al., 2017). You can download a tarball of this dataset from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz). The Waterbirds dataset can also be accessed through the [WILDS package](https://github.com/p-lambda/wilds), which will automatically download the dataset. Our code expects the following files/folders in the `datasets/cub` directory:

- `data/waterbird_complete95_forest2water2/`

First, we need to train the spurious environment predictor. A sample command to train assuming group annotations for whole validation set (set `val_frac` as 1) is :

`cd ./pseudo_labeling`

`python darp_fixmatch.py --name valfrac_1_wsampling_pseudobalance_imagenet_sgd_nolrdecay_lr_1e-3_wd_1e-4_epoch_100 --dataset cub_prev --image_size 224 --split semi --val_frac 1 --sampling group_weight --pseudo_balance --model resnet50 --pretrained imagenet --num_classes 2 --epochs 100 --optimizer sgd --lr 1e-3 --weight-decay 1e-4 --batch_size 64`



After training the model, we use the best saved model to generate the group probabilities (using `./robust_training/generate_pseudo_labels.py`) for all samples in the dataset. After this, we train a PG-DRO model using generated probabilties. The following code assumes that the generated probabilities are saved in `./robust_training/pseudo_waterbirds_valfrac_1.npy`. A sample command to run PG-DRO on Waterbirds is:

`cd ./robust_training`

`python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 1e-05 --batch_size 128 --weight_decay 1 --model resnet50 --n_epochs 300 --reweight_groups --robust --gamma 0.1 --generalization_adjustment 2 --allow_test --pseudo_label pseudo_waterbirds_valfrac_1`

### CelebA

Our code expects the following files/folders in the `datasets/celebA` directory:

- `data/list_eval_partition.csv`
- `data/list_attr_celeba.csv`
- `data/img_align_celeba/`

You can download these dataset files from [this Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset). The original dataset, due to Liu et al. (2015), can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The version of the CelebA dataset that we use in the paper (with the (hair, gender) groups) can also be accessed through the [WILDS package](https://github.com/p-lambda/wilds), which will automatically download the dataset.


First, we need to train the spurious environment predictor. A sample command to train on CelebA assuming group annotations for 5% of validation set (set `val_frac` as 0.05) is:

`cd ./pseudo_labeling`

`python darp_fixmatch.py --name valfrac_1_wsampling_pseudobalance_imagenet_sgd_nolrdecay_lr_1e-4_wd_1e-1_epoch_300 --dataset celeba --image_size 64 --split semi --val_frac 0.05 --sampling group_weight --pseudo_balance --model resnet50 --pretrained imagenet --num_classes 2 --epochs 300 --optimizer sgd --lr 1e-4 --weight-decay 1e-1 --batch_size 64 --seed 3`


After training the model, we use the best saved model to generate the group probabilities for all samples in the dataset. After this, we train a PG-DRO model using generated probabilties. A sample command to run PG-DRO on CelebA is:

`cd ./robust_training`

`python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.1 --lr 1e-05 --batch_size 128 --n_epochs 50 --save_step 1000 --save_best --save_last --reweight_groups --robust --alpha 0.01 --gamma 0.1 --allow_test --generalization_adjustment 1 --pseudo_label pseudo_celeba_valfrac_0.05`


### MultiNLI

Our code expects the following files/folders in the `datasets/multinli` directory:

- `data/metadata_random.csv`
- `glue_data/MNLI/cached_dev_bert-base-uncased_128_mnli`
- `glue_data/MNLI/cached_dev_bert-base-uncased_128_mnli-mm`
- `glue_data/MNLI/cached_train_bert-base-uncased_128_mnli`

These files can be downloaded easily from [GDRO](https://github.com/kohpangwei/group_DRO). 
Next, we train the spurious environment predictor. A sample command to train on MultiNLI is:

`cd ./pseudo_labeling`

`python darp_fixmatch.py --name valfrac_1_wsamling_pseudobalance_tau_0.95_class2_bert_lr_2e-5_wd_0_epoch_50 --dataset mnli_new --split semi --val_frac 1 --sampling group_weight --pseudo_balance --tau 0.95 --model bert  --num_classes 2 --epochs 50 --lr 2e-5 --weight-decay 0 --batch_size 32`

Next, we train a PG-DRO model using generated probabilties. A sample command to run PG-DRO on MultiNLI is:

`cd ./robust_training`

`python run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr 2e-05 --batch_size 32 --n_epochs 3 --save_step 1000 --save_best --save_last --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --allow_test --pseudo_label pseudo_multinli_valfrac_1` 

### CivilComments-WILDS

Our code expects the following files/folders in the `datasets/civilcomments_v1.0` directory:

- `data/all_data_with_identities.csv`
- `data/all_data_with_identities_backtrans.csv` (Generated using back translation)
  
Finally, we train the spurious environment predictor. A sample command to train on CivilComments-WILDS is:

`cd ./pseudo_labeling`

`python darp_fixmatch.py --name val_frac_1_wsamling_pseudobalance_tau_0.95_class2_bert_lr_1e-5_wd_1e-2_epoch_50 --dataset jigsaw --bias_name identity_any --split semi --val_frac 1 --sampling group_weight --pseudo_balance --tau 0.95 --model bert  --num_classes 2 --epochs 50 --lr 1e-5 --weight-decay 1e-2 --batch_size 8 --val_iteration 400`

Next, we train a PG-DRO model using generated probabilties. A sample command to run PG-DRO on MultiNLI is:

`cd ./robust_training`

`python run_expt.py -s confounder -d CivilComments -t toxicity -c identity_any --model bert --weight_decay 0.01 --lr 1e-05 --batch_size 16 --n_epochs 5 --save_step 1000 --save_best --save_last --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 1 --allow_test --pseudo_label pseudo_civil_valfrac_1`

## References
This codebase is adapted from [GDRO](https://github.com/kohpangwei/group_DRO) and [SSA](https://openreview.net/attachment?id=_F9xpOrqyX9&name=supplementary_material).
