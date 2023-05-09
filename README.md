# Distributionally Robust Optimization with Probabilistic Group
This codebase provides a Pytorch implementation for the AAAI 2023 Oral paper:

> Soumya Suvra Ghosal, and Yixuan Li
>
> [Distributionally Robust Optimization with Probabilistic Group](https://arxiv.org/abs/2303.05809)

The experiments use the following image-classification datasets:
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Waterbirds, formed from [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html) + [Places](http://places2.csail.mit.edu/)

<!-- - [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) -->

## Abstract
Modern machine learning models may be susceptible to learning spurious correlations that hold on average but not for the atypical group of samples. To address the problem, previous approaches minimize the empirical worst-group risk. Despite the promise, they often assume that each sample belongs to one and only one group, which does not allow expressing the uncertainty in group labeling. In this paper, we propose a novel framework PG-DRO, which explores the idea of probabilistic group membership for distributionally robust optimization. Key to our framework, we consider soft group membership instead of hard group annotations. The group probabilities can be flexibly generated using either supervised learning or zero-shot approaches. Our framework accommodates samples with group membership ambiguity, offering stronger flexibility and generality than the prior art. We comprehensively evaluate PG-DRO on both image classification and natural language processing benchmarks, establishing superior performance.

## Required Packages
Our experiments are conducted on Ubuntu Linux 20.04 with Python 3.9 and Pytorch 1.6. Besides, the following packages are required to be installed:
* Scipy
* Numpy
* Sklearn
* Pandas
* tqdm
* pillow
* torchvision
* pytorch_transformers

## Datasets

### WaterBirds
Similar to the construction in [Group_DRO](https://github.com/kohpangwei/group_DRO), this dataset is constructed by cropping out birds from photos in the Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011) and transferring them onto backgrounds from the Places dataset (Zhou et al., 2017). You can download a tarball of this dataset from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz). The Waterbirds dataset can also be accessed through the [WILDS package](https://github.com/p-lambda/wilds), which will automatically download the dataset. Our code expects the following files/folders in the `datasets/cub` directory:

- `data/waterbird_complete95_forest2water2/`

First, we need to train the spurious environment predictor.

After training the model, we use the best saved model to generate the group probabilities (using `./robust_training/generate_pseudo_labels.py`) for all samples in the dataset. After this, we train a PG-DRO model using generated probabilties. 

<!--The following code assumes that the generated probabilities are saved in `./robust_training/pseudo_waterbirds_valfrac_1.npy`. A sample command to run PG-DRO on Waterbirds is:

`cd ./robust_training`

`python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 1e-05 --batch_size 128 --weight_decay 1 --model resnet50 --n_epochs 300 --reweight_groups --robust --gamma 0.1 --generalization_adjustment 2 --allow_test --pseudo_label pseudo_waterbirds_valfrac_1` -->

  
## References
The codebase is adapted from [GDRO](https://github.com/kohpangwei/group_DRO).


## For bibtex citation 

```
@misc{ghosal2023distributionally,
      title={Distributionally Robust Optimization with Probabilistic Group}, 
      author={Soumya Suvra Ghosal and Yixuan Li},
      year={2023},
      eprint={2303.05809},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
