'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import optim
import numpy as np
from dataloader import BasicDataset
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import os
from cppimport import imp_from_filepath
from os.path import join, dirname
import dataloader
from pprint import pprint


class Sampling:
    def __init__(self, seed):
        try:
            path = join(dirname(__file__), "sources/sampling.cpp")
            self.sampling = imp_from_filepath(path)
            self.sampling.seed(seed)
            self.sample_ext = True
        except Exception as e:
            print(f'Exceptions "{e}" occured.')
            world.cprint("Cpp extension not loaded")
            self.sampling = None
            self.sample_ext = False

    def uniform_sample_original(self, dataset: BasicDataset, neg_ratio=1):
        """
        This method samples a user with a positive and a negative item (user, pos_item, neg_item).
        """
        if self.sample_ext:
            # Here we use a C++ library to do the sampling (0.1 seconds)
            s = self.sampling.sample_negative(
                dataset.n_user, dataset.m_item, dataset.train_data_size, dataset.all_pos, neg_ratio
            )
        else:
            # Here we use a python to do the sampling (9.7 seconds)
            s = self.uniform_sample_original_python(dataset)
        return s

    @staticmethod
    def uniform_sample_original_python(dataset: BasicDataset):
        """
        The original implimentation of BPR Sampling in LightGCN
        :return:
            np.array
        """
        user_num = dataset.train_data_size
        users = np.random.randint(0, dataset.n_user, user_num)
        all_pos = dataset.all_pos
        sample_list = []
        for i, user in enumerate(users):
            user_items = all_pos[user]
            if not user_items:
                continue
            posindex = np.random.randint(0, len(user_items))
            positem = user_items[posindex]
            while True:
                negitem = np.random.randint(0, dataset.m_item)
                if negitem not in user_items:
                    break
            sample_list.append([user, positem, negitem])
        return np.array(sample_list)


class BrpLoss:
    def __init__(
            self,
            rec_model: PairWiseModel,
            config: world.Config
    ):
        self.model = rec_model
        self.weight_decay = config.decay
        self.lr = config.lr
        self.opt = optim.Adam(rec_model.parameters(), lr=self.lr)

    def stage_one(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_file_name(config):

    if config.model_name == 'mf':
        file = f"mf-{config.dataset}-{config.latent_dim_rec}.pth.tar"
    elif config.model_name == 'lgn':
        file = f"lgn-{config.dataset}-{config.lightGCN_n_layers}-{config.latent_dim_rec}.pth.tar"
    else:
        raise NotImplementedError(f'getFileName does not have a path for the {config.model_name} model.')
    return os.path.join(config.file_path, file)


def train_minibatch(users, posItems, negItems, batch_size):
    for i in range(0, len(users), batch_size):
        yield tuple(x[i:i + batch_size] for x in (users, posItems, negItems))


def test_minibatch(users, batch_size):
    for i in range(0, len(users), batch_size):
        yield users[i:i + batch_size]


def shuffle(arrays, require_indices=False):
    """
    Shuffles the order of the samples in the arrays list. The arrays need to have the same length. Supports a single
    array in the arrays list.
    """

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class Timer:
    """
    Time context manager for lgcn_code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(Timer.TAPE) > 1:
            return Timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in Timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = Timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in Timer.NAMED_TAPE.items():
                Timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                Timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            Timer.NAMED_TAPE[kwargs['name']] = Timer.NAMED_TAPE[
                kwargs['name']] if Timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or Timer.TAPE

    def __enter__(self):
        self.start = Timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            Timer.NAMED_TAPE[self.named] += Timer.time() - self.start
        else:
            self.tape.append(Timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def recall_precision_at_k(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def mrr_at_k(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def ndcg_at_k_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def get_auc_score(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_item,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def get_label(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================



def load_dataset(config):
    if config.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        return dataloader.Loader(config)
    elif config.dataset == 'lastfm':
        return dataloader.LastFM(config)
    else:
        raise ValueError(f'Dataset {config.dataset} not supported!')


def print_config_info(config):
    attribute = [
        'bpr_batch_size', 'latent_dim_rec', 'lightGCN_n_layers', 'dropout', 'keep_prob', 'a_fold',
        'test_u_batch_size', 'multicore', 'lr', 'decay', 'pretrain', 'a_split', 'bigdata'
    ]
    config_dic = {a: getattr(config, a) for a in attribute}
    print('===========config================')
    pprint(config_dic)
    print("cores for test:", config.cores)
    print("comment:", config.comment)
    print("tensorboard:", config.tensorboard)
    print("LOAD:", config.load_bool)
    print("Weight path:", config.weight_path)
    print("Test Topks:", config.topks)
    print("using bpr loss")
    print('===========end===================')