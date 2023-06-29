

## Sampling Strategies for Enhanced Recommendation Performance: Advancements in LightGCN

This repository has for goal to reproduce the results from the following paper:

>SIGIR 2020. Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Authors: Alexandru Turcu, Bogdan Palfi, and Ryan Amaudruz

In additon, we investigate whether sampling strategies can benefit the model prediction performance.

## Abstract

Graph Convolutional Networks (GCNs) have seen increased popularity in the context of collaborative filtering based Recommender Systems. LightGCN is a heavily simplified variant of GCN, specifically designed for recommendation tasks. While at the time of publishing, LightGCN was able to achieve state-of-the-art performance, there was still room for improvement in the sampling mechanism of the loss function. This paper replicates the results of the original LightGCN paper and tests the model on a new dataset to further demonstrate its robustness. Furthermore, additional sampling methods are developed and tested on all datasets, some of which show increased performance in terms of recall or catalog coverage.

## Introduction

In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN,including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering

## Enviroment Requirement

`conda env create -f environment.yml`

## Dataset

There are 3 original datasets from the original repo: Gowalla, Yelp2018 and Amazon-book.
We added a new dataset: LastFM.

see more in `dataloader.py`

## An example to run a 3-layer LightGCN

run LightGCN on **Gowalla** dataset:

* command

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64`

## Summary of the changes made to the original repo
| File                           | Changes                                                             |
|--------------------------------|---------------------------------------------------------------------|
| dataloader.py                  | Refactored the code to improve performance and increase flexibility |
| Initial data exploration.ipynb | New file analyzing the datasets                                     |
| Item analysis.ipynb            | New file analyzing of model performance by item popularity          |
| main.py                        | Refactored the code for new functionality                           |
| model.py                       | Mostly unchanged                                                    |
| parse.py                       | Mostly unchanged                                                    |
| plotting.py                    | New file containing plotting functionality                          |
| procedure.py                   | Significant refactoring to manage sampling strategies               |
| utils.py                       | Significant refactoring to manage sampling strategies               |
| world.py                       | Minor changes                                                       |
| data                           | New dataset LastFM added                                            |
| results                        | Contains the test set performance                                   |
| checkpoints                    | Contains the best checkpoint from each model run                    |

In addition, detailed comments were added to the entire repository.