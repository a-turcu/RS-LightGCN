import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
import argparse

from model import BasicModel


def prepare_dir(file_path):
    """
    This function is used to create the directories needed to output a path. If the directories already exist, the
    function continues.
    """
    # Remove the file name to only keep the directory path.
    dir_path = '/'.join(file_path.split('/')[:-1])
    # Try to create the directory. Will have no effect if the directory already exists.
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass


class UltraGcn(BasicModel):
    def __init__(self, _, dataset, cache_path='./data/'):
        super().__init__()

        self.beta_uD = None
        self.beta_iD = None
        self.device = dataset.device

        self.user_num = dataset.n_user
        self.item_num = dataset.m_item
        self.embedding_dim = 64
        self.w1 = 1e-6
        self.w2 = 1
        self.w3 = 1e-6
        self.w4 = 1

        self.negative_weight = 300
        self.gamma = 1e-4
        self.lambda_ = 5e-4

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = dataset.user_item_net

        ii_cons_mat_path  = cache_path + 'ii_cons_mat.pickle'
        ii_neigh_mat_path = cache_path + 'ii_neigh_mat.pickle'

        prepare_dir(ii_cons_mat_path)

        self.calculation_initial(self.constraint_mat)

        if os.path.exists(ii_cons_mat_path):
            ii_constraint_mat = self.pload(ii_cons_mat_path)
            ii_neighbor_mat = self.pload(ii_neigh_mat_path)
        else:
            ii_constraint_mat, ii_neighbor_mat = self.get_ii_constraint_mat()
            self.pstore(ii_constraint_mat, ii_cons_mat_path)
            self.pstore(ii_neighbor_mat, ii_neigh_mat_path)

        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat

        self.initial_weight = 1e-4
        self.initial_weights()

    @staticmethod
    def pload(path):
        with open(path, 'rb') as f:
            res = pickle.load(f)
        print('load path = {} object'.format(path))
        return res

    @staticmethod
    def pstore(x, path):
        with open(path, 'wb') as f:
            pickle.dump(x, f)
        print('store object in path = {} ok'.format(path))

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

    def get_omegas(self, users, pos_items, neg_items):

        if self.w2 > 0:
            pos_weight = torch.mul(self.beta_uD[users], self.self.beta_iD[pos_items]).to(self.device)
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(self.device)

        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(
                torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)),
                self.constraint_mat['beta_iD'][neg_items.flatten()]
            ).to(self.device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(self.device)

        weight = torch.cat((pow_weight, neg_weight))
        return weight

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)  # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)  # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(self.device)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, neg_labels,weight=omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none'
        ).mean(dim=-1)

        pos_labels = torch.ones(pos_scores.size()).to(self.device)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, pos_labels, weight=omega_weight[:len(pos_scores)], reduction='none'
        )

        loss = pos_loss + neg_loss * self.negative_weight

        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        neighbor_embeds = self.item_embeds(
            self.ii_neighbor_mat[pos_items].to(self.device))  # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(self.device)  # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)

        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def test_foward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)

        return user_embeds.mm(item_embeds.t())

    def get_users_rating(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
        return user_embeds.mm(item_embeds.t())


    # def computer(self, users):
    #     """
    #     propagate methods for lightGCN
    #     """
    #     items = torch.arange(self.item_num).to(users.device)
    #     user_embeds = self.user_embeds(users)
    #     item_embeds = self.item_embeds(items)

    # def get_device(self):
    #     return self.user_embeds.weight.device

    def calculation_initial(self, train_mat, ii_diagonal_zero=False):
        print('Computing \\Omega for the item-item graph... ')
        self.A = train_mat.T.dot(train_mat)  # I * I
        self.n_items = self.A.shape[0]

        if ii_diagonal_zero:
            self.A[range(self.n_items), range(self.n_items)] = 0
        items_D = np.sum(self.A, axis=0).reshape(-1)
        users_D = np.sum(self.A, axis=1).reshape(-1)

        self.beta_uD = torch.asarray(np.sqrt(users_D + 1) / users_D.reshape(-1, 1))
        self.beta_iD = torch.asarray((1 / np.sqrt(items_D + 1)).reshape(1, -1))

    def get_ii_constraint_mat(self, num_neighbors=10):
        res_mat = torch.zeros((self.n_items, num_neighbors))
        res_sim_mat = torch.zeros((self.n_items, num_neighbors))
        all_ii_constraint_mat = torch.from_numpy(self.beta_uD.dot(self.beta_iD))
        for i in range(self.n_items):
            row = all_ii_constraint_mat[i] * torch.from_numpy(self.A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims
            if i % 1000 == 0:
                print('i-i constraint matrix {} ok'.format(i))

        print('Computation \\Omega OK!')
        return res_mat.long(), res_sim_mat.float()