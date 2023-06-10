"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
from os.path import join
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
        self.user_item_net = {}
        self.train_data_size = None
        self.n_user = None
        self.m_item = None
        self.test_dict = None
        self.all_pos = None

    def get_user_item_feedback(self, users, items):
        raise NotImplementedError

    def create_dataset_tensors(self, user_item_map):
        raise NotImplementedError

    def get_user_pos_items(self, users):
        return [self.user_item_net[user].nonzero()[1] for user in users]

    def get_sparse_graph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class LastFM(BasicDataset):
    """
    Dataset type for pytorch.
    Includes graph information.
    LastFM dataset.
    """

    def __init__(self, config, data_path='../data/lastfm'):
        super().__init__()
        # train or test
        cprint("loading [last fm]")
        self.device = config.device
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 1892
        self.m_item = 4489
        train_data = pd.read_table(join(data_path, 'data1.txt'), header=None)
        test_data = pd.read_table(join(data_path, 'test1.txt'), header=None)
        trust_net = pd.read_table(join(data_path, 'trustnetwork.txt'), header=None).to_numpy()
        trust_net -= 1
        train_data -= 1
        test_data -= 1
        self.trust_net = trust_net
        self.train_data = train_data
        self.test_data = test_data
        self.train_user = np.array(train_data[:][0])
        self.train_data_size = len(self.train_user)
        self.train_unique_users = np.unique(self.train_user)
        self.train_item = np.array(train_data[:][1])
        self.test_user = np.array(test_data[:][0])
        self.test_unique_users = np.unique(self.test_user)
        self.test_item = np.array(test_data[:][1])
        self.graph = None
        print(f"LastFm Sparsity : {(len(self.train_user) + len(self.test_user)) / self.n_user / self.m_item}")

        # (users,users)
        self.social_net = csr_matrix(
            (np.ones(len(trust_net)), (trust_net[:, 0], trust_net[:, 1])),
            shape=(self.n_user, self.n_user)
        )
        # (users,items), bipartite graph
        self.user_item_net = csr_matrix(
            (np.ones(len(self.train_user)), (self.train_user, self.train_item)),
            shape=(self.n_user, self.m_item)
        )

        # pre-calculate
        self.all_pos = self.get_user_pos_items(list(range(self.n_user)))
        self.all_neg = []
        all_items = set(range(self.m_item))
        for i in range(self.n_user):
            pos = set(self.all_pos[i])
            neg = all_items - pos
            self.all_neg.append(np.array(list(neg)))
        self.test_dict = self.__build_test()

    def get_sparse_graph(self):
        if self.graph is None:
            user_dim = torch.LongTensor(self.train_user)
            item_dim = torch.LongTensor(self.train_item)

            first_sub = torch.stack([user_dim, item_dim + self.n_user])
            second_sub = torch.stack([item_dim + self.n_user, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.graph = torch.sparse.IntTensor(
                index, data, torch.Size([self.n_user + self.m_item, self.n_user + self.m_item])
            )
            dense = self.graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.graph = torch.sparse.FloatTensor(
                index.t(), data, torch.Size([self.n_user + self.m_item, self.n_user + self.m_item])
            )
            self.graph = self.graph.coalesce().to(self.device)
        return self.graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def get_user_item_feedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.user_item_net[users, items]).astype('uint8').reshape((-1,))

    def __getitem__(self, index):
        user = self.train_unique_users[index]
        # return user_id and the positive items of the user
        return user

    def switch_2_test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.train_unique_users)


class Loader(BasicDataset):
    """
    Dataset type for pytorch. Includes graph information.
    """

    def __init__(self, config, minimal_bool=False):
        super().__init__()
        # train or test
        data_path = f'../data/{config.dataset}'
        cprint(f'loading [{data_path}]')
        self.split = config.a_split
        self.folds = config.a_fold
        self.device = config.device
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = data_path + '/train.txt'
        test_file = data_path + '/test.txt'
        self.path = data_path
        self.graph = None

        # Train data loading
        self.df_train = self.load_data_file(data_file=train_file)

        # Test data loading
        self.df_test = self.load_data_file(data_file=test_file)

        # Get user item info
        self.get_df_stats()

        # For some uses, we do not need the full init method.
        if minimal_bool:
            return

        # Print log
        self.print_dataset_info(config)

        # (users,items), bipartite graph
        self.user_item_net = csr_matrix(
            (np.ones(len(self.df_train['user_id'])), (self.df_train['user_id'], self.df_train['item_id'])),
            shape=(self.n_user, self.m_item)
        )
        self.users_D = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self.all_pos = self.get_user_pos_items(list(range(self.n_user)))
        self.test_dict = self.__build_test()
        print(f"{config.dataset} is ready to go")

    def get_df_stats(self):
        user_id_max = np.max((self.df_train['user_id'].max(), self.df_test['user_id'].max()))
        item_id_max = np.max((self.df_train['item_id'].max(), self.df_test['item_id'].max()))
        self.n_user = user_id_max + 1
        self.m_item = item_id_max + 1
        self.train_data_size = self.df_train.shape[0]
        self.test_data_size = self.df_test.shape[0]

    def print_dataset_info(self, config):
        print(f"{self.train_data_size} interactions for training")
        print(f"{self.test_data_size} interactions for testing")
        sparsity = (self.train_data_size + self.test_data_size) / self.n_user / self.m_item
        print(f"{config.dataset} Sparsity : {sparsity}")

    @staticmethod
    def load_data_file(data_file):
        """
        This method reads the train or test set text file and returns a tuple with information about the dataset.

        In the text file, each line first contains the user_id followed by the list of items the user_id interacted with.
        data_unique_users contains the user_ids in a list.
        data_user, data_item are lists that contain all the user_id item pairs.
        data_size is the length of the data_user, data_item lists.
        """
        with open(data_file) as f:
            # Loop through the lines
            lines = [line for line in f.readlines() if len(line) > 0]
        data_list = []
        for line in lines:
            line_split = line.strip('\n').split(' ')
            user_id = int(line_split[0])
            for item_id in line_split[1:]:
                if item_id: # Filters out cases when the user id has no associated items.
                    data_list.append({
                        'user_id': user_id,
                        'item_id': int(item_id)
                    })
        return pd.DataFrame(data_list)

    def create_dataset_tensors(self, user_item_map):
        # Create list to store the unique users, and the user item pairs.
        data_unique_users, data_item, data_user = [], [], []
        # Loop through the data file lines
        for user_id, user_items in user_item_map.items():
            # Store the user id
            data_unique_users.append(user_id)
            # Add extend the user list with the user id for each item
            data_user.extend([user_id] * len(user_items))
            # Add the items to the item list
            data_item.extend(user_items)
            # Track the max item id
            self.m_item = max(self.m_item, max(user_items))
            # Track the max user id
            self.n_user = max(self.n_user, user_id)
        return np.array(data_unique_users), np.array(data_user), np.array(data_item)

    def _split_a_hat(self, a_hat):
        A_fold = []
        fold_len = (self.n_user + self.m_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_user + self.m_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(a_hat[start:end]).coalesce().to(self.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        """
        This method creates a sparse matrix of size (n_user + n_item) x (n_user + n_item).
        This matrix is normalised and symmetric.
        """
        print("loading adjacency matrix")
        if self.graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except FileNotFoundError:
                print("generating adjacency matrix")
                s = time()
                mat_len = self.n_user + self.m_item
                adj_mat = sp.dok_matrix((mat_len, mat_len), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.user_item_net.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()

                d_inv = np.power(np.array(adj_mat.sum(axis=1)), -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                # This represents "A tilde" the original paper.
                norm_adj = (d_mat @ adj_mat @ d_mat).tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split:
                self.graph = self._split_a_hat(norm_adj)
                print("done split matrix")
            else:
                self.graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.graph = self.graph.coalesce().to(self.device)
                print("don't split the matrix")
        return self.graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for _, row in self.df_test.iterrows():
            if row.user_id in test_data:
                test_data[row.user_id].append(row.item_id)
            else:
                test_data[row.user_id] = [row.item_id]
        return test_data

    def get_user_item_feedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.user_item_net[users, items]).astype('uint8').reshape((-1,))
