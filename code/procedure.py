"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
"""


import numpy as np
import torch
import utils
from utils import Timer
import multiprocessing


class ProcedureManager:
    def __init__(self, config):
        self.sampler_helper = utils.Sampling(config.seed)
        self.device = config.device
        self.topks = config.topks
        self.tensorboard = config.tensorboard
        self.bpr_batch_size = config.bpr_batch_size
        self.test_u_batch_size = config.test_u_batch_size

    def bpr_train_original(self, dataset, model, loss, epoch, w=None, config=None):
        model.train()

        with Timer(name="Sample"):
            s = self.sampler_helper.UniformSample_original(dataset)
        users = torch.Tensor(s[:, 0]).long()
        pos_items = torch.Tensor(s[:, 1]).long()
        neg_items = torch.Tensor(s[:, 2]).long()

        users = users.to(config.device)
        pos_items = pos_items.to(config.device)
        neg_items = neg_items.to(config.device)
        users, pos_items, neg_items = utils.shuffle((users, pos_items, neg_items))
        total_batch = len(users) // config.bpr_batch_size + 1
        aver_loss = 0.
        enumerator = enumerate(utils.train_minibatch(users, pos_items, neg_items, batch_size=self.bpr_batch_size))
        for batch_i, (batch_users, batch_pos, batch_neg) in enumerator:
            cri = loss.stage_one(batch_users, batch_pos, batch_neg)
            aver_loss += cri
            if config.tensorboard:
                w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / self.bpr_batch_size) + batch_i)
        aver_loss = aver_loss / total_batch
        time_info = Timer.dict()
        Timer.zero()
        return f"loss{aver_loss:.3f}-{time_info}"

    def test_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = utils.get_label(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        for k in self.topks:
            ret = utils.recall_precision_at_k(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(utils.ndcg_at_k_r(groundTrue, r, k))
        return {
            'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg)
        }


    def test(self, dataset, model, epoch, w=None, multicore=0):
        u_batch_size = self.test_u_batch_size
        dataset: utils.BasicDataset
        testDict: dict = dataset.test_dict
        # eval mode with no dropout
        model = model.eval()
        max_K = max(self.topks)
        pool = None
        if multicore == 1:
            cores = multiprocessing.cpu_count() // 2
            pool = multiprocessing.Pool(cores)
        results = {'precision': np.zeros(len(self.topks)),
                   'recall': np.zeros(len(self.topks)),
                   'ndcg': np.zeros(len(self.topks))}
        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.test_minibatch(users, batch_size=u_batch_size):
                allPos = dataset.get_user_pos_items(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(self.device)

                rating = model.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [
                #         utils.AUC(rating[i],
                #                   dataset,
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.test_one_batch(x))
            scale = float(u_batch_size/len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if self.tensorboard:
                w.add_scalars(
                    f'Test/Recall@{self.topks}',
                    {str(self.topks[i]): results['recall'][i] for i in range(len(self.topks))}, epoch
                )
                w.add_scalars(
                    f'Test/Precision@{self.topks}',
                    {str(self.topks[i]): results['precision'][i] for i in range(len(self.topks))}, epoch
                )
                w.add_scalars(
                    f'Test/NDCG@{self.topks}',
                    {str(self.topks[i]): results['ndcg'][i] for i in range(len(self.topks))}, epoch
                )
            if multicore == 1:
                pool.close()
            print(results)
            return results
