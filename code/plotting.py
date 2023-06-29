# Module import
from model import LightGCN
from utils import test_minibatch, prepare_dir
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


def get_config(dataset):
    """
    This function receives a dataset string and prepares a config object that stores all the config parameters.

    New function
    """
    # def run_analysis_and_plot(dataset):
    # Instantiate the arguments for the config file
    args = FakeArgs()

    # Set the dataset to load
    args.dataset = dataset

    # Return the config
    return Config(
        args.dataset, args.model, args.bpr_batch, args.recdim, args.layer, args.dropout, args.keepprob, args.a_fold,
        args.testbatch, args.multicore, args.lr, args.decay, args.pretrain, args.seed, args.epochs, args.load,
        args.checkpoint_path, args.results_path, args.topks, args.tensorboard, args.comment, args.sampling
    )


def get_test_results(config, data_loader, file_path):
    """
    New function based on original code
    """
    # Instantiate the recommender system model
    rec_model = LightGCN(config, data_loader)

    # Move the model to the device
    rec_model = rec_model.to(config.device)

    # Load the checkpoint file
    weight_file = 'checkpoints/' + file_path
    print(f"load and save to {weight_file}")
    # Safe model loading
    rec_model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
    print(f"loaded model weights from {weight_file}")

    # Eval mode
    rec_model = rec_model.eval()

    # Run evaluation using model checkpoint.
    u_batch_size = 100

    # Perform inference to get test metrics
    with torch.no_grad():
        # Get a list of the unique user ids
        users = list(data_loader.test_dict.keys())
        # Warning for batch size
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        # Create empty lists to store data
        rating_list = []
        groundTrue_list = []

        # Loop through the user batches
        for batch_users in test_minibatch(users, batch_size=u_batch_size):
            # Get the ids of the positive items in the train set for each user in the batch
            all_pos = data_loader.get_user_pos_items(batch_users)
            # Get the ids of the positive items in the test set for each user in the batch
            ground_true = [data_loader.test_dict[u] for u in batch_users]
            # Add the users to the GPU
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to('cuda')
            # Get all item ratings for each user in the batch
            rating = rec_model.get_users_rating(batch_users_gpu)
            # Exclude the positive items from the train set
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(all_pos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            # Select the top 20 items for each user in the batch
            rating_K = torch.topk(rating, k=20)[1]
            # Delete the ratings from memory
            rating = rating.cpu().numpy()
            del rating
            # Store the users, with the top k predictions and the ground truth values
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(ground_true)

    # Unnest the results
    ground_truth_list = []
    for sub_groundTrue_list in groundTrue_list:
        ground_truth_list += sub_groundTrue_list

    new_rating_list = []
    for r in rating_list:
        new_rating_list += r

    return ground_truth_list, new_rating_list


def count_tp_fn_fp(ground_truth_list, new_rating_list):
    """
    Counts the true positive, false positive and fase negative items for each user.

    ground_truth_list and new_rating_list are two lists that have the same length as
    the number of users and the users have for id the index of the list.

    New function
    """
    # Create a list to store each data piece
    data_list = []
    # For each user with their respective predicted items and ground truth
    for user_id, (rating, gt) in enumerate(zip(new_rating_list, ground_truth_list)):
        # Convert the predicted and ground truth item to sets.
        rating_set = set(rating.tolist())
        gt_set = set(gt)

        # Determine what items are TP, FP, FN
        true_pos = rating_set & gt_set
        false_pos = rating_set - gt_set
        false_neg = gt_set - rating_set

        # Store each instance of user item pair
        for item_id in true_pos:
            data_list.append({'user_id': user_id, 'item_id': item_id, 'pred_typ': 'tp'})

        for item_id in false_pos:
            data_list.append({'user_id': user_id, 'item_id': item_id, 'pred_typ': 'fp'})

        for item_id in false_neg:
            data_list.append({'user_id': user_id, 'item_id': item_id, 'pred_typ': 'fn'})

    # Return a dataframe with the information
    return pd.DataFrame(data_list)


def user_centric_analysis(df, user_to_item_count_map):
    """
    Analysing the results from the user's perspective.

    New function
    """
    # Grouped df by user id
    user_gr = df.groupby(['user_id', 'pred_typ'])['item_id'].count().reset_index()
    user_pivot = user_gr.pivot(index='user_id', columns='pred_typ', values='item_id').reset_index().fillna(0)

    # Retrieve the item count for each user.
    user_pivot['item_count'] = user_pivot['user_id'].map(user_to_item_count_map)

    # Create a categorical variable to bin the users by item count.
    user_pivot['item_count_gr'] = ''
    user_pivot.loc[user_pivot['item_count'] < 11, 'item_count_gr'] = 'less_11'
    user_pivot.loc[(user_pivot['item_count'] >= 11) & (user_pivot['item_count'] < 16), 'item_count_gr'] = '11_to_16'
    user_pivot.loc[(user_pivot['item_count'] >= 16) & (user_pivot['item_count'] < 30), 'item_count_gr'] = '16_to_30'
    user_pivot.loc[user_pivot['item_count'] >= 30, 'item_count_gr'] = 'more_30'

    # Group the users by item count bin
    agg_dic = {'user_id': 'count', 'fn': np.mean, 'fp': np.mean, 'tp': np.mean, 'item_count': np.mean}
    rename_dic = {'user_id': 'count'}
    user_pivot_gr = user_pivot.groupby('item_count_gr').agg(agg_dic).rename(columns=rename_dic).reset_index()
    user_pivot_gr = user_pivot_gr.sort_values('item_count').drop(columns='item_count').reset_index(drop=True)
    user_pivot_gr['precision'] = user_pivot_gr['tp'] / (user_pivot_gr['tp'] + user_pivot_gr['fp'])
    user_pivot_gr['recall'] = user_pivot_gr['tp'] / (user_pivot_gr['tp'] + user_pivot_gr['fn'])

    return user_pivot_gr


def item_centric_analysis(df, item_to_user_count_map):
    """
    Analysing the results from the item's perspective.

    New function
    """
    # Grouped df by item id
    item_gr = df.groupby(['item_id', 'pred_typ'])['user_id'].count().reset_index()
    item_pivot = item_gr.pivot(index='item_id', columns='pred_typ', values='user_id').reset_index().fillna(0)

    # Retrieve the user count for each item.
    item_pivot['user_count'] = item_pivot['item_id'].map(item_to_user_count_map)

    # Create a categorical variable to bin the items by user count.
    item_pivot['user_count_gr'] = ''
    item_pivot.loc[item_pivot['user_count'] < 10, 'user_count_gr'] = 'less_10'
    item_pivot.loc[(item_pivot['user_count'] >= 10) & (item_pivot['user_count'] < 13), 'user_count_gr'] = '10_to_13'
    item_pivot.loc[(item_pivot['user_count'] >= 13) & (item_pivot['user_count'] < 20), 'user_count_gr'] = '13_to_20'
    item_pivot.loc[item_pivot['user_count'] >= 20, 'user_count_gr'] = 'more_20'

    # Group the items by user count bin.
    agg_dic = {'item_id': 'count', 'fn': np.mean, 'fp': np.mean, 'tp': np.mean, 'user_count': np.mean}
    rename_dic = {'item_id': 'count'}
    item_pivot_gr = item_pivot.groupby('user_count_gr').agg(agg_dic).rename(columns=rename_dic).reset_index()
    item_pivot_gr = item_pivot_gr.sort_values('user_count').drop(columns='user_count').reset_index(drop=True)
    item_pivot_gr['precision'] = item_pivot_gr['tp'] / (item_pivot_gr['tp'] + item_pivot_gr['fp'])
    item_pivot_gr['recall'] = item_pivot_gr['tp'] / (item_pivot_gr['tp'] + item_pivot_gr['fn'])

    return item_pivot_gr


def create_item_centric_plot(df, task='save_fig'):
    """
    Function to plot the model outcome by item popularity.

    New function
    """
    # Rename the user count column for the plot
    user_count_gr_rename = {
        'less_10': 'Less than 10',
        '10_to_13': '10 to 12',
        '13_to_20': '13 to 19',
        'more_20': 'More than 19'
    }
    df['user_count_gr'] = df['user_count_gr'].map(user_count_gr_rename)
    # Update the column names
    col_rename = {
        'user_count_gr': 'Number of users',
        'precision': 'Precision',
        'recall': 'Recall',
    }
    df = df.rename(columns=col_rename)
    # Pivot the data to get the precision and recall for each user group
    df_plot_pivot = pd.melt(
        df, id_vars='Number of users', value_vars=['Precision', 'Recall'], var_name='Metrics', value_name='%'
    )
    # Convert to % and round
    df_plot_pivot['%'] = df_plot_pivot['%'].map(lambda x: np.round(x * 100, 1))
    # Bar plot creation
    ax = sns.barplot(data=df_plot_pivot, x='Number of users', y="%", hue="Metrics")
    # Add values to the bar
    for i in ax.containers:
        ax.bar_label(i, )

    if task == 'save_fig':
        out_path = f'plots/item_score_by_popularity_{f.split(".")[0]}.png'
        prepare_dir(out_path)
        plt.savefig(out_path)
        plt.clf()
    else:
        plt.show()

