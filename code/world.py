'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
import sys
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing


class Config:
    def __init__(self):
        all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
        all_models = ['mf', 'lgn']
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        args = parse_args()

        self.root_path = os.path.dirname(os.path.dirname(__file__))
        self.code_path = join(self.root_path, 'code')
        self.data_path = join(self.root_path, 'data')
        self.board_path = join(self.code_path, 'runs')
        self.file_path = join(self.code_path, 'checkpoints')

        sys.path.append(join(self.code_path, 'sources'))

        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path, exist_ok=True)

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
        self.A_split = False
        self.bigdata = False
        self.gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu else "cpu")
        self.cores = multiprocessing.cpu_count() // 2
        self.seed = args.seed

        self.dataset = args.dataset
        self.model_name = args.model
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

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
