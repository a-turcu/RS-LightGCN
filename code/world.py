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
        self.dataset = 'yelp2018' # 'gowalla'
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


class Config:
    def __init__(self):
        args = parse_args()
        # args = FakeArgs()
        self.file_path = FILE_PATH
        self.board_path = BOARD_PATH
        self.bpr_batch_size = args.bpr_batch
        self.latent_dim_rec = args.recdim
        self.lightGCN_n_layers = args.layer
        self.dropout = args.dropout
        self.keep_prob = args.keepprob
        self.a_fold= args.a_fold
        self.test_u_batch_size = args.testbatch
        self.multicore = args.multicore
        self.lr = args.lr
        self.decay = args.decay
        self.pretrain = args.pretrain
        self.a_split = False
        self.bigdata = False
        self.gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu else "cpu")
        self.cores = multiprocessing.cpu_count() // 2
        self.seed = args.seed
        self.dataset = args.dataset
        self.model_name = args.model

        all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
        all_models = ['mf', 'lgn']
        if self.dataset not in all_dataset:
            raise NotImplementedError(f"Haven't supported {self.dataset} yet!, try {all_dataset}")
        if self.model_name not in all_models:
            raise NotImplementedError(f"Haven't supported {self.model_name} yet!, try {all_models}")

        self.train_epochs = args.epochs
        self.load_bool = args.load
        self.weight_path = args.path
        self.topks = eval(args.topks)
        self.tensorboard = args.tensorboard
        self.comment = args.comment
        # let pandas shut up
        from warnings import simplefilter
        simplefilter(action="ignore", category=FutureWarning)


def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
