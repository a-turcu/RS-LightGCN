import pandas as pd

from model import LightGCN
from utils import BrpLoss, test_minibatch
from world import FakeArgs, Config
from dataloader import DataLoader
import torch

import os

try:
    os.chdir('code')
except FileNotFoundError:
    pass

args = FakeArgs()

# Instantiate the config object
config = Config(
    args.dataset, args.model, args.bpr_batch, args.recdim, args.layer, args.dropout, args.keepprob, args.a_fold,
    args.testbatch, args.multicore, args.lr, args.decay, args.pretrain, args.seed, args.epochs, args.load,
    args.checkpoint_path, args.results_path, args.topks, args.tensorboard, args.comment, args.sampling
)
dataset = DataLoader(config)
train_df = dataset.load_train_file()

# Instantiate the recommender system model
rec_model = LightGCN(config, dataset)
# Move the model to the device
rec_model = rec_model.to(config.device)
# Instantiate the BPRLoss
loss = BrpLoss(rec_model, config)
# # Load weight file
# weight_file = get_file_name(config)


#
# weight_file = 'checkpoints/lgn-gowalla-3-64.pth.tar'
#
# # weight_file = 'checkpoints/lgn-gowalla-3-64.pth.tar'

weight_file = 'checkpoints/lgn-gowalla-3-64-stratified_original.pth.tar'

print(f"load and save to {weight_file}")

rec_model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
print(f"loaded model weights from {weight_file}")
test_df = dataset.load_test_file()


# eval mode with no dropout
rec_model = rec_model.eval()

u_batch_size = 100

with torch.no_grad():
    users = list(dataset.test_dict.keys())
    try:
        assert u_batch_size <= len(users) / 10
    except AssertionError:
        print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
    users_list = []
    rating_list = []
    groundTrue_list = []
    top_ranked_list = []
    # auc_record = []
    # ratings = []
    total_batch = len(users) // u_batch_size + 1
    for batch_users in test_minibatch(users, batch_size=u_batch_size):
        all_pos = dataset.get_user_pos_items(batch_users)
        ground_true = [dataset.test_dict[u] for u in batch_users]
        batch_users_gpu = torch.Tensor(batch_users).long()
        batch_users_gpu = batch_users_gpu.to('cuda')

        rating = rec_model.get_users_rating(batch_users_gpu)
        # rating = rating.cpu()
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(all_pos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1 << 10)
        rating_K = torch.topk(rating, k=20)[1]

        rating = rating.cpu().numpy()
        # aucs = [
        #         utils.AUC(rating[i],
        #                   dataset,
        #                   test_data) for i, test_data in enumerate(ground_true)
        #     ]
        # auc_record.extend(aucs)
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(ground_true)

ground_truth_list = []
for sub_groundTrue_list in groundTrue_list:
    ground_truth_list += sub_groundTrue_list

new_rating_list = []
for r in rating_list:
    new_rating_list += r

new_user_list = []
for u in users_list:
    new_user_list += u

data_list = []
for user_id, (rating, gt) in enumerate(zip(new_rating_list, ground_truth_list)):
    rating_set = set(rating.tolist())
    gt_set = set(gt)

    true_pos = rating_set & gt_set
    false_pos = rating_set - gt_set
    false_neg = gt_set - rating_set

    for item_id in true_pos:
        data_list.append({'user_id': user_id, 'item_id': item_id, 'pred_typ': 'tp'})

    for item_id in false_pos:
        data_list.append({'user_id': user_id, 'item_id': item_id, 'pred_typ': 'fp'})

    for item_id in false_neg:
        data_list.append({'user_id': user_id, 'item_id': item_id, 'pred_typ': 'fn'})

import pandas as pd
df_analysis = pd.DataFrame(data_list)

type_map = {
    'tp': 1,
    'fp': 0,
    'fn': 0,
}
df_analysis['is_correct'] = df_analysis['pred_typ'].map(type_map)

user_gr = df_analysis.groupby(['user_id', 'pred_typ', 'is_correct'])['item_id'].count().reset_index()
user_pivot = user_gr.pivot(index='user_id', columns='pred_typ', values='item_id').reset_index().fillna(0)

item_gr = df_analysis.groupby(['item_id', 'pred_typ', 'is_correct'])['user_id'].count().reset_index()
item_pivot = item_gr.pivot(index='item_id', columns='pred_typ', values='user_id').reset_index().fillna(0)


train_df_u_gr = train_df.groupby('user_id')['item_id'].count().reset_index()
train_df_u_sort = train_df_u_gr.sort_values('item_id').reset_index(drop=True)
train_df_u_sort['cur_index'] = train_df_u_sort.index

user_to_itemcountmap = dict(zip(train_df_u_sort['user_id'], train_df_u_sort['item_id']))


train_df_i_gr = train_df.groupby('item_id')['user_id'].count().reset_index()
train_df_i_sort = train_df_i_gr.sort_values('user_id').reset_index(drop=True)

item_to_user_countmap = dict(zip(train_df_i_sort['item_id'], train_df_i_sort['user_id']))

user_pivot['item_count'] = user_pivot['user_id'].map(user_to_itemcountmap)

user_pivot['item_count_gr'] = ''
user_pivot.loc[user_pivot['item_count'] < 12, 'item_count_gr'] = 'less_12'
user_pivot.loc[(user_pivot['item_count'] >= 12) & (user_pivot['item_count'] < 20), 'item_count_gr'] = '12_to_20'
user_pivot.loc[(user_pivot['item_count'] >= 20) & (user_pivot['item_count'] < 50), 'item_count_gr'] = '20_to_50'
user_pivot.loc[user_pivot['item_count'] >= 50, 'item_count_gr'] = 'more_50'

user_pivot_gr = user_pivot.groupby('item_count_gr')[['fn', 'fp', 'tp']].mean().reset_index()
user_pivot_gr['precision'] = user_pivot_gr['tp'] / (user_pivot_gr['tp'] + user_pivot_gr['fp'])
user_pivot_gr['recall'] = user_pivot_gr['tp'] / (user_pivot_gr['tp'] + user_pivot_gr['fn'])




item_pivot['user_count'] = item_pivot['item_id'].map(item_to_user_countmap)

item_pivot['user_count_gr'] = ''
item_pivot.loc[item_pivot['user_count'] < 10, 'user_count_gr'] = 'less_10'
item_pivot.loc[(item_pivot['user_count'] >= 10) & (item_pivot['user_count'] < 14), 'user_count_gr'] = '10_to_14'
item_pivot.loc[(item_pivot['user_count'] >= 14) & (item_pivot['user_count'] < 25), 'user_count_gr'] = '14_to_25'
item_pivot.loc[item_pivot['user_count'] >= 25, 'user_count_gr'] = 'more_25'


item_pivot[item_pivot['tp'] == 0]['user_count'].mean()
item_pivot[item_pivot['tp'] > 3]['user_count'].mean()

item_pivot_gr = item_pivot.groupby('user_count_gr')[['fn', 'fp', 'tp']].mean().reset_index()
item_pivot_gr['precision'] = item_pivot_gr['tp'] / (item_pivot_gr['tp'] + item_pivot_gr['fp'])
item_pivot_gr['recall'] = item_pivot_gr['tp'] / (item_pivot_gr['tp'] + item_pivot_gr['fn'])



item_pivot['fp_minus_fn'] = item_pivot['fp'] - item_pivot['fn']

item_pivot['fp_minus_fn'].quantile(0.4)

item_pivot['fp_minus_fn_gr'] = ''
item_pivot.loc[item_pivot['fp_minus_fn'] < -4, 'item_count_gr'] = 'less_-3'
item_pivot.loc[(item_pivot['item_count'] >= -4) & (item_pivot['item_count'] < -2), 'item_count_gr'] = '12_to_20'
item_pivot.loc[(item_pivot['item_count'] >= -2) & (item_pivot['item_count'] < 0), 'item_count_gr'] = '20_to_50'
item_pivot.loc[item_pivot['item_count'] >= 0, 'item_count_gr'] = 'more_50'


# ((user_pivot['item_count'] >= 12) & (user_pivot['item_count'] < 20)).sum()
# ((user_pivot['item_count'] >= 20) & (user_pivot['item_count'] < 50)).sum()
# (user_pivot['item_count'] >= 50).sum()






#
#
#
#
# import matplotlib.pyplot as plt
#
# import seaborn as sns
#
# sns.lineplot(data=train_df_u_sort, x='cur_index', y='item_id')
# plt.show()
#
# import matplotlib.pyplot as plt
#
# import seaborn as sns
#
# sns.lineplot(data=train_df_u_sort.iloc[:10_000], x='cur_index', y='item_id')
# plt.show()
#
#
# train_df_i_gr = train_df.groupby('item_id')['user_id'].count().reset_index()
# train_df_i_sort = train_df_i_gr.sort_values('user_id').reset_index(drop=True)
#
# train_df_i_sort['cur_index'] = train_df_i_sort.index
#
# import matplotlib.pyplot as plt
#
# import seaborn as sns
#
# sns.lineplot(data=train_df_i_sort, x='cur_index', y='user_id')
# plt.show()
#
#
#
#
