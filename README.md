# Distributionally Robust Optimization with Probabilistic Group
This codebase provides a Pytorch implementation for the AAAI 2023 Oral paper:

> Soumya Suvra Ghosal, and Yixuan Li
>
> [Distributionally Robust Optimization with Probabilistic Group]()

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

TO BE ADDED
  
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
