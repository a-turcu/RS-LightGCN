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
from dataloader import DataLoader
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import os
from cppimport import imp_from_filepath
from os.path import join, dirname
import dataloader
from pprint import pprint
import pandas as pd


class Sampling:
    def __init__(self, seed, sampling):
        self.top_ranked_items = None
        self.epoch = None
        self.pos_count = None
        self.neg_count = None
        self.m_item = None
        self.weights_norm = None
        try:
            path = join(dirname(__file__), "sources/sampling.cpp")
            self.sampling = imp_from_filepath(path)
            self.sampling.seed(seed)
            self.sample_ext = True
        except Exception as e:
            print(f'Exceptions "{e}" occurred.')
            world.cprint("Cpp extension not loaded")
            self.sampling = None
            self.sample_ext = False

        sample_map = {
            'original': self.uniform_sample_original,
            'new_random': self.new_random_sample,
            'hard_neg': self.hard_neg_sample,
            'hard_neg2': self.hard_neg_sample2,
            'hard_neg3': self.hard_neg_sample3,
            'hard_neg4': self.hard_neg_sample4,
            'stratified_original': self.stratified_sample_original,
            'stratified_new_random': self.stratified_random_sample,
            'normalised_sample_original': self.normalised_sample_original,
            'weigthed_item_prob_sampling': self.weigthed_item_prob_sampling,
            'stronger_weigthed_item_prob_sampling': self.stronger_weigthed_item_prob_sampling,
        }
        if sampling in sample_map:
            self.sample = sample_map[sampling]
        else:
            raise ValueError(f'Sampling method {sampling} is not supported!')

    def uniform_sample_original(self, dataset: DataLoader, neg_ratio=1):
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
    def uniform_sample_original_python(dataset: DataLoader):
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
            if not len(user_items):
                continue
            posindex = np.random.randint(0, len(user_items))
            positem = user_items[posindex]
            while True:
                negitem = np.random.randint(0, dataset.m_item)
                if negitem not in user_items:
                    break
            sample_list.append([user, positem, negitem])
        return np.array(sample_list)

    @staticmethod
    def new_random_sample(dataset):
        """
        The new random sampling respects the original distribution of the positive item and simply selects randomly
        the negative items.
        """
        sample_list = []
        for user_id, item_id in zip(dataset.df_train['user_id'], dataset.df_train['item_id']):
            while True:
                neg_item = np.random.randint(0, dataset.m_item)
                if neg_item not in dataset.all_pos_map[user_id]:
                    break
            sample_list.append([user_id, item_id, neg_item])
        return np.array(sample_list)

    @staticmethod
    def stratified_random_sample(dataset):
        """
        This is a stratified random strategy, where both item selected should have a similar popularity.
        """
        sample_list = []
        for user_id, item_id in zip(dataset.df_train['user_id'], dataset.df_train['item_id']):
            strat_gr = dataset.item_id_to_strat_gr_map[item_id]
            item_list_len = dataset.strat_gr_to_item_list_len[strat_gr]
            while True:
                neg_item_index = np.random.randint(0, item_list_len)
                neg_item = dataset.strat_gr_to_item_list[strat_gr][neg_item_index]
                if neg_item not in dataset.all_pos_map[user_id]:
                    break
            sample_list.append([user_id, item_id, neg_item])
        return np.array(sample_list)

    @staticmethod
    def stratified_sample_original(dataset):
        """
        The original sampling strategy stratified by item popularity
        """
        sample_list = []
        for user_id in dataset.df_train['user_id'].unique():
            user_items = dataset.all_pos[user_id]
            user_items_len = len(user_items)
            for _ in range(dataset.mean_item_per_user):
                pos_index = np.random.randint(0, user_items_len)
                pos_item = user_items[pos_index]
                strat_gr = dataset.item_id_to_strat_gr_map[pos_item]
                item_list_len = dataset.strat_gr_to_item_list_len[strat_gr]
                while True:
                    neg_item_index = np.random.randint(0, item_list_len)
                    neg_item = dataset.strat_gr_to_item_list[strat_gr][neg_item_index]
                    if neg_item not in dataset.all_pos_map[user_id]:
                        break
                sample_list.append([user_id, pos_item, neg_item])
        return np.array(sample_list)

    def normalised_sample_original(self, dataset):
        """
        In this function, we try to "normalise" the number of items that are being randomly selected such that
        there is a more equal number of items selected as positive and negative on each run.
        """
        if self.pos_count is None:
            self.pos_count = {item_id: 0 for item_id in dataset.df_train['item_id'].unique()}
            self.neg_count = self.pos_count.copy()
        elif self.epoch > 10:
            # Create a pool with the items that have been less selected as negatives.
            median_neg_count = np.median([v for v in self.neg_count.values()])
            neg_item_pool = [k for k, v in self.neg_count.items() if v <= median_neg_count]
            neg_item_pool_len = len(neg_item_pool)

            # Create a weight map for the pos item weighing.
            median_pos_count = np.median([v for v in self.pos_count.values()])
            pos_weight_map = {k: 2 if v <= median_pos_count else 1 for k, v in self.pos_count.items()}

        sample_list = []
        if self.epoch > 10:
            for user_id in dataset.df_train['user_id'].unique():
                # Retrieve the positive objects for the user
                user_items_raw = dataset.all_pos[user_id].tolist()
                # Create a list that contains the positive items that have a low count
                items_with_low_pos_count = [i for i in dataset.all_pos[user_id] if pos_weight_map[i] == 2]
                # Combine both list
                user_items = user_items_raw + items_with_low_pos_count
                user_items_len = len(user_items)
                for i in range(dataset.mean_item_per_user):
                    # Randomly select index
                    pos_index = np.random.randint(0, user_items_len)
                    pos_item = user_items[pos_index]
                    # Break when we find a positive item
                    while True:
                        # We only sample for the items that were selected as negative less.
                        neg_item = neg_item_pool[np.random.randint(0, neg_item_pool_len)]
                        if neg_item not in user_items:
                            break
                    # Counting
                    self.pos_count[pos_item] += 1
                    self.neg_count[neg_item] += 1
                    # Add sample
                    sample_list.append([user_id, pos_item, neg_item])
        else:
            # Original random sampling
            for user_id in dataset.df_train['user_id'].unique():
                user_items = dataset.all_pos[user_id]
                user_items_len = len(user_items)
                for i in range(dataset.mean_item_per_user):
                    pos_index = np.random.randint(0, user_items_len)
                    pos_item = user_items[pos_index]
                    while True:
                        neg_item = np.random.randint(0, dataset.m_item)
                        if neg_item not in user_items:
                            break

                    self.pos_count[pos_item] += 1
                    self.neg_count[neg_item] += 1
                    sample_list.append([user_id, pos_item, neg_item])
        return np.array(sample_list)

    def weigthed_item_prob_sampling(self, dataset):
        """
        In this function, we want to balance out the items by popularity. Meaning that if an item is popular, it
        will be more likely to be chosen as a positive item. Hence, we add a bit more probably for less popular items
        to be selected as positives.
        """
        arr_len = 2*dataset.mean_item_per_user
        if self.weights_norm is None:
            item_gr = dataset.df_train.groupby('item_id')['user_id'].count().reset_index()
            median = item_gr['user_id'].median()
            item_gr['weight'] = 1
            cond = item_gr['user_id'] <= median
            item_gr.loc[cond, 'weight'] = (median / item_gr.loc[cond, 'user_id'])**0.5
            item_to_weight_dic = dict(zip(item_gr['item_id'], item_gr['weight']))

            self.weights_norm = []
            all_pos_adj = []
            for i, item_list in enumerate(dataset.all_pos):
                weights = [item_to_weight_dic[i] for i in item_list]
                weights_sum = sum(weights)
                weights_norm = [w / weights_sum for w in weights]
                all_pos_adj.append(np.random.choice(item_list, size=arr_len, p=weights_norm))
                self.weights_norm.append(weights_norm)
        else:
            all_pos_adj = []
            for i, item_list in enumerate(dataset.all_pos):
                all_pos_adj.append(np.random.choice(item_list, size=arr_len, p=self.weights_norm[i]))

        sample_list = []
        # Original random sampling
        for user_id in dataset.df_train['user_id'].unique():
            user_items = dataset.all_pos[user_id]
            for i in range(dataset.mean_item_per_user):
                pos_item = all_pos_adj[user_id][np.random.randint(0, arr_len)]
                while True:
                    neg_item = np.random.randint(0, dataset.m_item)
                    if neg_item not in user_items:
                        break
                sample_list.append([user_id, pos_item, neg_item])
        return np.array(sample_list)

    def stronger_weigthed_item_prob_sampling(self, dataset):
        """
        In this function, we want to balance out the items by popularity. Meaning that if an item is popular, it
        will be more likely to be chosen as a positive item. Hence, we add a bit more probably for less popular items
        to be selected as positives.
        """
        arr_len = 2 * dataset.mean_item_per_user
        if self.weights_norm is None:
            item_gr = dataset.df_train.groupby('item_id')['user_id'].count().reset_index()
            median = item_gr['user_id'].median()
            item_gr['weight'] = 1
            cond = item_gr['user_id'] <= median
            item_gr.loc[cond, 'weight'] = (median / item_gr.loc[cond, 'user_id'])**0.5
            item_gr['weight'] = (item_gr['weight'] - 1) * 2 + 1
            item_to_weight_dic = dict(zip(item_gr['item_id'], item_gr['weight']))

            self.weights_norm = []
            all_pos_adj = []
            for i, item_list in enumerate(dataset.all_pos):
                weights = [item_to_weight_dic[i] for i in item_list]
                weights_sum = sum(weights)
                weights_norm = [w / weights_sum for w in weights]
                all_pos_adj.append(np.random.choice(item_list, size=arr_len, p=weights_norm))
                self.weights_norm.append(weights_norm)
        else:
            all_pos_adj = []
            for i, item_list in enumerate(dataset.all_pos):
                all_pos_adj.append(np.random.choice(item_list, size=arr_len, p=self.weights_norm[i]))

        sample_list = []
        # Original random sampling
        for user_id in dataset.df_train['user_id'].unique():
            user_items = dataset.all_pos[user_id]
            for i in range(dataset.mean_item_per_user):
                pos_item = all_pos_adj[user_id][np.random.randint(0, arr_len)]
                while True:
                    neg_item = np.random.randint(0, dataset.m_item)
                    if neg_item not in user_items:
                        break
                sample_list.append([user_id, pos_item, neg_item])
        return np.array(sample_list)

    def hard_neg_sample(self, dataset):
        """
        This is a hard sampling strategy
        """
        if self.epoch < 50:
            return self.new_random_sample(dataset)
        sample_list = []
        for user_id, item_id in zip(dataset.df_train['user_id'], dataset.df_train['item_id']):
            rand_int = np.random.randint(0, 1000)
            neg_item = int(self.top_ranked_items[user_id, rand_int])
            sample_list.append([user_id, item_id, neg_item])
        return np.array(sample_list)

    def hard_neg_sample2(self, dataset, hard_neg_prob=0.5):
        """
        This sampling method hard samples only half the time
        """
        if self.epoch < 50:
            return self.new_random_sample(dataset)
        sample_list = []
        for user_id, item_id in zip(dataset.df_train['user_id'], dataset.df_train['item_id']):
            if np.random.rand() < hard_neg_prob:
                # Hard neg sampling
                rand_int = np.random.randint(0, 1000)
                neg_item = int(self.top_ranked_items[user_id, rand_int])
            else:
                # Random sampling
                while True:
                    neg_item = np.random.randint(0, dataset.m_item)
                    if neg_item not in dataset.all_pos_map[user_id]:
                        break
            sample_list.append([user_id, item_id, neg_item])
        return np.array(sample_list)

    def hard_neg_sample3(self, dataset, hard_neg_prob=0.01):
        """
        This sampling method hard samples only half the time
        """
        if self.epoch < 5:
            return self.new_random_sample(dataset)
        sample_list = []
        # Original random sampling
        for user_id in dataset.df_train['user_id'].unique():
            user_items = dataset.all_pos[user_id]
            arr_len = len(user_items)
            for i in range(dataset.mean_item_per_user):
                pos_item = user_items[np.random.randint(0, arr_len)]
                if np.random.rand() < hard_neg_prob:
                    # Hard neg sampling
                    rand_int = np.random.randint(0, 1000)
                    neg_item = int(self.top_ranked_items[user_id, rand_int])
                else:
                    # Random sampling
                    while True:
                        neg_item = np.random.randint(0, dataset.m_item)
                        if neg_item not in dataset.all_pos_map[user_id]:
                            break
                sample_list.append([user_id, pos_item, neg_item])
        return np.array(sample_list)

    def hard_neg_sample4(self, dataset, hard_neg_prob=0.05):
        """
        This sampling method hard samples only half the time
        """
        if self.epoch < 5:
            return self.new_random_sample(dataset)
        sample_list = []
        # Original random sampling
        for user_id in dataset.df_train['user_id'].unique():
            user_items = dataset.all_pos[user_id]
            arr_len = len(user_items)
            for i in range(dataset.mean_item_per_user):
                pos_item = user_items[np.random.randint(0, arr_len)]
                if np.random.rand() < hard_neg_prob:
                    # Hard neg sampling
                    rand_int = np.random.randint(0, 1000)
                    neg_item = int(self.top_ranked_items[user_id, rand_int])
                else:
                    # Random sampling
                    while True:
                        neg_item = np.random.randint(0, dataset.m_item)
                        if neg_item not in dataset.all_pos_map[user_id]:
                            break
                sample_list.append([user_id, pos_item, neg_item])
        return np.array(sample_list)

    def mixed(self, dataset, original_prob=0.5):
        """
        This sampling method mixes the original sampling and random sampling strategy.
        """
        if np.random.rand() < original_prob:
            return self.uniform_sample_original(dataset)
        else:
            return self.new_random_sample(dataset)


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


def get_file_name_base(config):
    if config.model_name == 'mf':
        return f"mf-{config.dataset}-{config.latent_dim_rec}-{config.sampling}"
    elif config.model_name == 'lgn':
        return f"lgn-{config.dataset}-{config.lightGCN_n_layers}-{config.latent_dim_rec}-{config.sampling}"
    else:
        raise NotImplementedError(f'getFileName does not have a path for the {config.model_name} model.')


def get_checkpoint_file_name(config):
    return os.path.join(config.checkpoint_path, get_file_name_base(config) + '.pth.tar')


def get_results_file_name(config):
    return os.path.join(config.results_path, get_file_name_base(config) + '.csv')


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
    print("Weight path:", config.checkpoint_path)
    print("Test Topks:", config.topks)
    print("using bpr loss")
    print('===========end===================')