import dataloader
from pprint import pprint


def load_dataset(config):
    if config.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        return dataloader.Loader(config)
    elif config.dataset == 'lastfm':
        return dataloader.LastFM()
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
