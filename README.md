# Distributionally Robust Optimization with Probabilistic Group
This codebase provides a Pytorch implementation for the AAAI 2023 Oral paper: Distributionally Robust Optimization with Probabilistic Group.

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
* timm
* pytorch_transformers

## Datasets

In our study, we use the following challenging benchmarks :
  - WaterBirds:  Similar to the construction in [Group_DRO](https://github.com/kohpangwei/group_DRO), this dataset is constructed by cropping out birds from photos in the Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011) and transferring them onto backgrounds from the Places dataset (Zhou et al., 2017).
  - [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): Large-scale CelebFaces Attributes Dataset. 
  
  
## References
The codebase is adapted from [GDRO](https://github.com/kohpangwei/group_DRO).


## For bibtex citation 

```

```
