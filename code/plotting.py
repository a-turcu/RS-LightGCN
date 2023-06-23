# Module import
from model import LightGCN
from utils import BrpLoss, test_minibatch
from world import FakeArgs, Config
from dataloader import DataLoader
import torch
import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the current dir is 'code'
try:
    os.chdir('code')
except FileNotFoundError:
    pass

# Instantiate the arguments for the config file
args = FakeArgs()

# Instantiate the config object
config = Config(
    args.dataset, args.model, args.bpr_batch, args.recdim, args.layer, args.dropout, args.keepprob, args.a_fold,
    args.testbatch, args.multicore, args.lr, args.decay, args.pretrain, args.seed, args.epochs, args.load,
    args.checkpoint_path, args.results_path, args.topks, args.tensorboard, args.comment, args.sampling
)
dataset = DataLoader(config)
train_df = dataset.load_train_file()
n_items = train_df['item_id'].unique().shape[0]

# Instantiate the recommender system model
rec_model = LightGCN(config, dataset)
# Move the model to the device
rec_model = rec_model.to(config.device)

# Obtain a user to item count map
train_df_u_gr = train_df.groupby('user_id')['item_id'].count().reset_index()
train_df_u_sort = train_df_u_gr.sort_values('item_id').reset_index(drop=True)
user_to_item_count_map = dict(zip(train_df_u_sort['user_id'], train_df_u_sort['item_id']))

# Obtain an item to user count map
train_df_i_gr = train_df.groupby('item_id')['user_id'].count().reset_index()
train_df_i_sort = train_df_i_gr.sort_values('user_id').reset_index(drop=True)
item_to_user_count_map = dict(zip(train_df_i_sort['item_id'], train_df_i_sort['user_id']))

plotting = True

# Create dictionary to store information for each model checkpoint.
df_analysis = {}
user_pivot = {}
item_pivot = {}
catalogue_coverage = {}
user_item_pivot = {}
user_pivot_gr = {}
item_pivot_gr = {}
ax_dic = {}

# Loop through the checkpoint files
file_list = [f for f in os.listdir('checkpoints/') if f.endswith('.pth.tar')]
for f in file_list:
    # Record time
    start = time.time()
    # Load the checkpoint file
    weight_file = 'checkpoints/' + f
    print(f"load and save to {weight_file}")
    try:
        rec_model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        print(f"loaded model weights from {weight_file}")
    except:
        print(f'Checkpoint file {f} could not be loaded!')

    # Load the test dataframe
    test_df = dataset.load_test_file()

    # Eval mode
    rec_model = rec_model.eval()

    # Run evaluation using model checkpoint.
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
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(ground_true)

    # Unnest the results
    ground_truth_list = []
    for sub_groundTrue_list in groundTrue_list:
        ground_truth_list += sub_groundTrue_list

    new_rating_list = []
    for r in rating_list:
        new_rating_list += r

    new_user_list = []
    for u in users_list:
        new_user_list += u

    catalogue_coverage[f] = len(set(torch.concat(new_rating_list).tolist())) / n_items

    # Create a dataframe with the results for each user item pair
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

    df_analysis[f] = pd.DataFrame(data_list)

    # Grouped df by user id
    user_gr = df_analysis[f].groupby(['user_id', 'pred_typ'])['item_id'].count().reset_index()
    user_pivot[f] = user_gr.pivot(index='user_id', columns='pred_typ', values='item_id').reset_index().fillna(0)

    # Grouped df by item id
    item_gr = df_analysis[f].groupby(['item_id', 'pred_typ'])['user_id'].count().reset_index()
    item_pivot[f] = item_gr.pivot(index='item_id', columns='pred_typ', values='user_id').reset_index().fillna(0)

    # Retrieve the item count for each user.
    user_pivot[f]['item_count'] = user_pivot[f]['user_id'].map(user_to_item_count_map)

    # Create a categorical variable to bin the users by item count.
    user_pivot[f]['item_count_gr'] = ''
    user_pivot[f].loc[user_pivot[f]['item_count'] < 11, 'item_count_gr'] = 'less_11'
    user_pivot[f].loc[(user_pivot[f]['item_count'] >= 11) & (user_pivot[f]['item_count'] < 16), 'item_count_gr'] = '11_to_16'
    user_pivot[f].loc[(user_pivot[f]['item_count'] >= 16) & (user_pivot[f]['item_count'] < 30), 'item_count_gr'] = '16_to_30'
    user_pivot[f].loc[user_pivot[f]['item_count'] >= 30, 'item_count_gr'] = 'more_30'

    # Group the users by item count bin
    agg_dic = {'user_id': 'count', 'fn': np.mean, 'fp': np.mean, 'tp': np.mean, 'item_count': np.mean}
    rename_dic = {'user_id': 'count'}
    user_pivot_gr[f] = user_pivot[f].groupby('item_count_gr').agg(agg_dic).rename(columns=rename_dic).reset_index()
    user_pivot_gr[f] = user_pivot_gr[f].sort_values('item_count').drop(columns='item_count').reset_index(drop=True)
    user_pivot_gr[f]['precision'] = user_pivot_gr[f]['tp'] / (user_pivot_gr[f]['tp'] + user_pivot_gr[f]['fp'])
    user_pivot_gr[f]['recall'] = user_pivot_gr[f]['tp'] / (user_pivot_gr[f]['tp'] + user_pivot_gr[f]['fn'])

    # Retrieve the user count for each item.
    item_pivot[f]['user_count'] = item_pivot[f]['item_id'].map(item_to_user_count_map)

    # Create a categorical variable to bin the items by user count.
    item_pivot[f]['user_count_gr'] = ''
    item_pivot[f].loc[item_pivot[f]['user_count'] < 10, 'user_count_gr'] = 'less_10'
    item_pivot[f].loc[(item_pivot[f]['user_count'] >= 10) & (item_pivot[f]['user_count'] < 13), 'user_count_gr'] = '10_to_13'
    item_pivot[f].loc[(item_pivot[f]['user_count'] >= 13) & (item_pivot[f]['user_count'] < 20), 'user_count_gr'] = '13_to_20'
    item_pivot[f].loc[item_pivot[f]['user_count'] >= 20, 'user_count_gr'] = 'more_20'

    # Group the items by user count bin.
    agg_dic = {'item_id': 'count', 'fn': np.mean, 'fp': np.mean, 'tp': np.mean, 'user_count': np.mean}
    rename_dic = {'item_id': 'count'}
    item_pivot_gr[f] = item_pivot[f].groupby('user_count_gr').agg(agg_dic).rename(columns=rename_dic).reset_index()
    item_pivot_gr[f] = item_pivot_gr[f].sort_values('user_count').drop(columns='user_count').reset_index(drop=True)
    item_pivot_gr[f]['precision'] = item_pivot_gr[f]['tp'] / (item_pivot_gr[f]['tp'] + item_pivot_gr[f]['fp'])
    item_pivot_gr[f]['recall'] = item_pivot_gr[f]['tp'] / (item_pivot_gr[f]['tp'] + item_pivot_gr[f]['fn'])

    # Retrieve the item count for each user.
    df_analysis[f]['item_count'] = df_analysis[f]['user_id'].map(user_to_item_count_map)
    # Retrieve the user count for each item.
    df_analysis[f]['user_count'] = df_analysis[f]['item_id'].map(item_to_user_count_map)

    # Create a categorical variable to bin the users by item count.
    df_analysis[f]['item_count_gr'] = ''
    df_analysis[f].loc[df_analysis[f]['item_count'] < 11, 'item_count_gr'] = 'less_11'
    df_analysis[f].loc[(df_analysis[f]['item_count'] >= 11) & (df_analysis[f]['item_count'] < 16), 'item_count_gr'] = '11_to_16'
    df_analysis[f].loc[(df_analysis[f]['item_count'] >= 16) & (df_analysis[f]['item_count'] < 30), 'item_count_gr'] = '16_to_30'
    df_analysis[f].loc[df_analysis[f]['item_count'] >= 30, 'item_count_gr'] = 'more_30'

    # Create a categorical variable to bin the items by user count.
    df_analysis[f]['user_count_gr'] = ''
    df_analysis[f].loc[df_analysis[f]['user_count'] < 10, 'user_count_gr'] = 'less_10'
    df_analysis[f].loc[(df_analysis[f]['user_count'] >= 10) & (df_analysis[f]['user_count'] < 13), 'user_count_gr'] = '10_to_13'
    df_analysis[f].loc[(df_analysis[f]['user_count'] >= 13) & (df_analysis[f]['user_count'] < 20), 'user_count_gr'] = '13_to_20'
    df_analysis[f].loc[df_analysis[f]['user_count'] >= 20, 'user_count_gr'] = 'more_20'

    # Grouped df by user id
    user_item_gr = df_analysis[f].groupby(['item_count_gr', 'user_count_gr', 'pred_typ'])['item_id'].count().reset_index()

    # Pivot the dataframe
    user_item_pivot[f] = user_item_gr.pivot(
        index=['user_count_gr', 'item_count_gr'],
        columns='pred_typ',
        values='item_id'
    ).reset_index().fillna(0)

    # Sort the rows
    user_count_gr_sort_map = {'less_10': 1, '10_to_13': 2, '13_to_20': 3, 'more_20': 4}
    item_count_gr_sort_map = {'less_11': 1, '11_to_16': 2, '16_to_30': 3, 'more_30': 4}
    user_item_pivot[f]['user_order'] = user_item_pivot[f]['user_count_gr'].map(user_count_gr_sort_map)
    user_item_pivot[f]['item_order'] = user_item_pivot[f]['item_count_gr'].map(item_count_gr_sort_map)
    sorting_columns = ['user_order', 'item_order']
    user_item_pivot[f] = user_item_pivot[f].sort_values(sorting_columns).drop(columns=sorting_columns).reset_index(drop=True)

    user_item_pivot[f]['count'] = user_item_pivot[f]['tp'] + user_item_pivot[f]['fn']
    user_item_pivot[f]['precision'] = user_item_pivot[f]['tp'] / (user_item_pivot[f]['tp'] + user_item_pivot[f]['fp'])
    user_item_pivot[f]['recall'] = user_item_pivot[f]['tp'] / (user_item_pivot[f]['tp'] + user_item_pivot[f]['fn'])

    df_plot = item_pivot_gr[f].copy()
    user_count_gr_rename = {
        'less_10': 'Less than 10',
        '10_to_13': '10 to 12',
        '13_to_20': '13 to 19',
        'more_20': 'More than 19'
    }
    df_plot['user_count_gr'] = df_plot['user_count_gr'].map(user_count_gr_rename)

    col_rename = {
        'user_count_gr': 'Number of users',
        'precision': 'Precision',
        'recall': 'Recall',
    }
    df_plot = df_plot.rename(columns=col_rename)
    df_plot = df_plot.rename(columns={'user_count_gr': 'Number of users'})
    df_plot_pivot = pd.melt(
        df_plot, id_vars='Number of users', value_vars=['Precision', 'Recall'], var_name='Metrics', value_name='%'
    )

    df_plot_pivot['%'] = df_plot_pivot['%'].map(lambda x: np.round(x * 100, 1))
    ax_dic[f] = sns.barplot(data=df_plot_pivot, x='Number of users', y="%", hue="Metrics")
    for i in ax_dic[f].containers:
        ax_dic[f].bar_label(i, )
    plt.title(f'Items grouped by popularity - {f.split(".")[0].split("-")[-1]}')
    plt.savefig(f'plot-{f.split(".")[0]}.png')
    plt.clf()

    # Time recording printout.
    print(f'File {f} took: {time.time() - start}')



# train_df_u_sort['cur_index'] = train_df_u_sort.index
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


#
# # df10 = item_pivot_gr['lgn-gowalla-3-64-original.pth.tar']
# #
# #
# # df10.to_csv('df10.csv', index=False)
#
# df_plot = pd.read_csv('df10.csv')
#
#
# df11 = pd.melt(df_plot, id_vars='item_count_gr', value_vars=['fn', 'fp', 'tp'])
#
#
#
# df_plot_pivot = pd.melt(df_plot, id_vars='item_count_gr', value_vars=['precision', 'recall'])
#
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# # sns.barplot(data=df11, x="item_count_gr", y="value", hue="variable")
#
#
# sns.barplot(data=df_plot_pivot, x="item_count_gr", y="value", hue="variable")
# plt.show()
#
#
# # # set width of bar
# barWidth = 0.25
# fig = plt.subplots(figsize=(12, 8))
#
# # set height of bar
#
#
# # Set position of bar on X axis
# br1 = np.arange(len(df10))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
#
# # Make the plot
# plt.bar(br1, df10['fn'], color ='r', width = barWidth,
# 		edgecolor ='grey', label ='fn')
# plt.bar(br2, df10['fp'], color ='g', width = barWidth,
# 		edgecolor ='grey', label ='fp')
# plt.bar(br3, df10['tp'], color ='b', width = barWidth,
# 		edgecolor ='grey', label ='tp')
#
# # Adding Xticks
# plt.xlabel('Branch', fontweight ='bold', fontsize = 15)
# plt.ylabel('Students passed', fontweight ='bold', fontsize = 15)
# # plt.xticks([r + barWidth for r in range(len(df10))],
# # 		['2015', '2016', '2017', '2018', '2019'])
# #
# # plt.legend()
# plt.show()
#
#

