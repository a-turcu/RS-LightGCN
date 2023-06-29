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
from world import cprint
from time import time


class DataLoader(Dataset):
    """
    This class handles the data loading and preprocessing.
    """
    def __init__(self, config, minimal_bool=False):
        super().__init__()
        # train or test
        self.data_path = f'../data/{config.dataset}'
        cprint(f'loading [{self.data_path}]')
        self.dataset = config.dataset
        self.split = config.a_split
        self.folds = config.a_fold
        self.device = config.device
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        self.path = self.data_path
        self.graph = None
        self.train_data_size = None
        self.test_data_size = None
        self.all_pos = None
        self.all_pos_map = None
        self.all_items = None
        self.all_pos_list_map = None
        self.test_dict = None
        self.item_id_to_popularity_map = None
        self.item_id_to_strat_gr_map = None
        self.strat_gr_to_item_list = None
        # Train data loading
        self.df_train = self.load_train_file()
        # Test data loading
        self.df_test = self.load_test_file()
        # Preprocess the data
        self.df_train, self.df_test = self.data_preprocessing(self.df_train, self.df_test)
        # Get user item info
        self.get_df_stats()
        # For some uses, we do not need the full init method.
        if minimal_bool:
            return
        # Print log
        self.print_dataset_info(config)
        # Define the graph
        self.graph_definition()
        # Pre-calculations
        self.pre_calculation()
        # Create a dictionary to store the test

        print(f"{config.dataset} is ready to go")

    def load_train_file(self):
        """
        This method load the training data file as a pandas dataframe.
        """
        return self.load_data_file(self.data_path + '/train.txt')

    def load_test_file(self):
        """
        This method load the test data file as a pandas dataframe.
        """
        return self.load_data_file(self.data_path + '/test.txt')

    def data_preprocessing(self, train_data, test_data):
        """
        This method ensures that the train and test sets have the same user and item ids. It prints out information
        relating to the changes made.
        In practice, only the lastfm dataset requires this type of cleaning.
        """
        total_train_dropped = 0
        total_test_dropped = 0
        user_id_changed = False
        item_id_changed = False

        while True:
            train_data, test_data, train_len_diff2, test_len_diff2, user_id_change = self.clean_dataframe(
                train_data, test_data, 'user_id'
            )
            train_data, test_data, train_len_diff1, test_len_diff1, item_id_change = self.clean_dataframe(
                train_data, test_data, 'item_id'
            )
            user_id_changed = user_id_changed or user_id_change
            item_id_changed = item_id_changed or item_id_change
            train_len_diff = train_len_diff1 + train_len_diff2
            test_len_diff = test_len_diff1 + test_len_diff2
            total_train_dropped += train_len_diff
            total_test_dropped += test_len_diff
            if (train_len_diff == 0) and (test_len_diff == 0):
                break
        print(
            f'{total_train_dropped} training samples and {total_test_dropped} test samples were dropped during the data'
            ' cleaning.'
        )
        if user_id_changed:
            print('The user ids were updated.')
        else:
            print('The user ids were not updated.')
        if item_id_changed:
            print('The item ids were updated.')
        else:
            print('The item ids were not updated.')
        return train_data, test_data

    @staticmethod
    def clean_dataframe(train_data, test_data, col_to_clean):
        """
        This method ensures that the elements in the col_to_clean column present in the test set are present in the
        training. It then ensures that the id have a sequential ordering that starts at index 0.
        """
        common_ids = list(set(train_data[col_to_clean]))
        common_ids.sort()
        new_train_data = train_data[train_data[col_to_clean].isin(common_ids)].reset_index(drop=True)
        new_test_data = test_data[test_data[col_to_clean].isin(common_ids)].reset_index(drop=True)
        id_map = {old_uid: new_uid for new_uid, old_uid in enumerate(common_ids)}
        id_change = bool(sum([k != v for k, v in id_map.items()]))
        if id_change:
            new_train_data[col_to_clean] = new_train_data[col_to_clean].map(id_map)
            new_test_data[col_to_clean] = new_test_data[col_to_clean].map(id_map)
        train_len_diff = train_data.shape[0] - new_train_data.shape[0]
        test_len_diff = test_data.shape[0] - new_test_data.shape[0]
        columns = new_train_data.columns.tolist()
        return (
            new_train_data.sort_values(columns).reset_index(drop=True),
            new_test_data.sort_values(columns).reset_index(drop=True),
            train_len_diff,
            test_len_diff,
            id_change
        )

    def pre_calculation(self):
        self.all_pos = self.get_user_pos_items(list(range(self.n_user)))
        self.all_pos_map = {user_id: set(pos_list) for user_id, pos_list in enumerate(self.all_pos)}
        self.all_items = set(self.df_train['item_id'].unique().tolist())
        self.all_pos_list_map = self.get_sorted_list_map(self.all_pos_map)
        self.test_dict = self.__build_test()
        train_df_i_gr = self.df_train.groupby('item_id')['user_id'].count().reset_index()
        quartile1 = train_df_i_gr['user_id'].quantile(0.25)
        quartile2 = train_df_i_gr['user_id'].quantile(0.5)
        quartile3 = train_df_i_gr['user_id'].quantile(0.75)
        train_df_i_gr['stratified_group'] = None
        cond = train_df_i_gr['user_id'] <= quartile1
        train_df_i_gr.loc[cond, 'stratified_group'] = 'quartile_1'
        cond = (train_df_i_gr['user_id'] > quartile1) & (train_df_i_gr['user_id'] <= quartile2)
        train_df_i_gr.loc[cond, 'stratified_group'] = 'quartile_2'
        cond = (train_df_i_gr['user_id'] > quartile2) & (train_df_i_gr['user_id'] <= quartile3)
        train_df_i_gr.loc[cond, 'stratified_group'] = 'quartile_3'
        cond = (train_df_i_gr['user_id'] > quartile3)
        train_df_i_gr.loc[cond, 'stratified_group'] = 'quartile_4'
        self.item_id_to_popularity_map = dict(zip(train_df_i_gr['item_id'], train_df_i_gr['user_id']))
        self.item_id_to_strat_gr_map = dict(zip(train_df_i_gr['item_id'], train_df_i_gr['stratified_group']))
        self.strat_gr_to_item_list = {}
        for k, v in self.item_id_to_strat_gr_map.items():
            if v in self.strat_gr_to_item_list:
                self.strat_gr_to_item_list[v].append(k)
            else:
                self.strat_gr_to_item_list[v] = [k]
        self.strat_gr_to_item_list_len = {k: len(v) for k, v in self.strat_gr_to_item_list.items()}
        self.mean_item_per_user = int(self.df_train.shape[0] / self.df_train.user_id.unique().shape[0])

    def get_sorted_list_map(self, set_map):
        """
        This method receives a dictionary that contains unsorted lists as values and returns a dictionary with sorted
        lists as values.
        """
        sorted_list_map = {}
        for k, v in set_map.items():
            item_list = list(v)
            item_list.sort()
            sorted_list_map[k] = item_list
        return sorted_list_map

    def graph_definition(self):
        """
        This function creates the spares matrix
        """
        # Bipartite graph
        self.user_item_net = csr_matrix(
            (np.ones(len(self.df_train['user_id'])), (self.df_train['user_id'], self.df_train['item_id'])),
            shape=(self.n_user, self.m_item)
        )
        self.users_D = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

    def get_df_stats(self):
        user_id_max = np.max((self.df_train['user_id'].max(), self.df_test['user_id'].max()))
        item_id_max = np.max((self.df_train['item_id'].max(), self.df_test['item_id'].max()))
        self.all_user = set(self.df_train['user_id']) | set(self.df_test['user_id'])
        self.all_item = set(self.df_train['item_id']) | set(self.df_test['item_id'])
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
                if item_id:  # Filters out cases when the user id has no associated items.
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
                pre_adj_mat = sp.load_npz(self.data_path + '/s_pre_adj_mat.npz')
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
                sp.save_npz(self.data_path + '/s_pre_adj_mat.npz', norm_adj)

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

    def get_user_pos_items(self, users):
        return [self.user_item_net[user].nonzero()[1] for user in users]
