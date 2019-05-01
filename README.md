# [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760)
by Xin Chen, [Lingxi Xie](http://lingxixie.com/), [Jun Wu](https://see.tongji.edu.cn/info/1153/6850.htm) and [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN).

This repository contains the search and evaluation code for our work Progressive DARTS.

It requires only **0.3 GPU-days** (7 hours on a single P100 card) to finish a search progress on CIFAR10 and CIFAR100 datasets,
much faster than DARTS, and achieves higher classification accuracy on both CIFAR and ImageNet datasets (mobole setting).

**This code is based on the implementation of  [DARTS](https://github.com/quark0/darts).**

# Introduction

TBD soon.

## Usage

To run our code, you need a GPU with at least **16GB memory**, and equip it with PyTorch 0.3 or above versions.

#### Run the following command to perform a search progress on CIFAR10.

```
python train_search.py \\
       --tmp_data_dir /path/to/your/data \\
       --save log_path \\
       --add_layers 6 \\
       --add_layers 12 \\
       --dropout_rate 0.1 \\
       --dropout_rate 0.4 \\
       --dropout_rate 0.7 \\
       --note note_of_this_run
Add --cifar100 if search on CIFAR100.
```

It needs ~7 hours on a single P100 GPU, or 12 hours on a single 1080-Ti GPU to finish everything.

For the parameters, please see our paper (we would provided more explanations in this README soon).

#### The evaluation process simply follows that of DARTS.

###### Here is the evaluation on CIFAR10/100:

```
python train_cifar.py \\
       --tmp_data_dir /path/to/your/data \\
       --save log_path \\
       --note note_of_this_run
Add --cifar100 if evaluating on CIFAR100.
```

###### Here is the evaluation on ImageNet (mobile setting):
```
python train_imagenet.py \\
       --tmp_data_dir /path/to/your/data \\
       --save log_path \\
       --auxiliary \\
       --note note_of_this_run
```
