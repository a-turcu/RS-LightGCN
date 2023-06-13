'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
import multiprocessing
import sys

from parse import parse_args

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')

sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


class FakeArgs:
    def __init__(self):
        self.a_fold = 100
        self.bpr_batch = 2048
        self.comment = 'lgn'
        self.dataset = 'lastfm'  # 'lastfm'  # '' #'gowalla'  # 'yelp2018'  # 'amazon-book'
        self.decay = 0.0001
        self.dropout = 0
        self.epochs = 1000 # 1000
        self.keepprob = 0.6
        self.layer = 3
        self.load = 0
        self.lr = 0.001
        self.model = 'lgn'
        self.multicore = 0
        self.path = './checkpoints'
        self.pretrain = 0
        self.recdim = 64
        self.seed = 2020
        self.tensorboard = 1
        self.testbatch = 100
        self.topks = '[20]'
        self.sampling = 'hard_neg'


class Config:
    def __init__(
            self, dataset, model, bpr_batch, recdim, layer, dropout, keepprob, a_fold, testbatch, multicore, lr=0.001,
            decay=0.0001, pretrain=0, seed=2020, epochs=1000, load=0, path='./checkpoints', topks='[20]', tensorboard=1,
            comment='lgn', sampling='original'
    ):
        import subprocess

        self.bpr_batch_size = bpr_batch
        self.latent_dim_rec = recdim
        self.lightGCN_n_layers = layer
        self.dropout = dropout
        self.keep_prob = keepprob
        self.a_fold= a_fold
        self.test_u_batch_size = testbatch
        self.multicore = multicore
        self.lr = lr
        self.decay = decay
        self.pretrain = pretrain
        self.seed = seed
        self.dataset = dataset
        self.model_name = model
        self.sampling = sampling
        self.file_path = FILE_PATH
        self.board_path = BOARD_PATH
        self.a_split = False
        self.bigdata = False
        self.gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu else "cpu")
        self.cores = multiprocessing.cpu_count() // 2
        self.all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
        self.all_models = ['mf', 'lgn']
        if self.dataset not in self.all_dataset:
            raise NotImplementedError(f"Haven't supported {self.dataset} yet!, try {self.all_dataset}")
        if self.model_name not in self.all_models:
            raise NotImplementedError(f"Haven't supported {self.model_name} yet!, try {self.all_models}")
        self.train_epochs = epochs
        self.load_bool = load
        self.weight_path = path
        self.topks = eval(topks)
        self.tensorboard = tensorboard
        self.comment = comment
        # Silence pandas
        from warnings import simplefilter
        simplefilter(action="ignore", category=FutureWarning)


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
