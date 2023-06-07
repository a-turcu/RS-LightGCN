from world import Config
import dataloader
import model
import utils
from pprint import pprint


def load_dataset(config):
    if config.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        return dataloader.Loader(config)
    elif config.dataset == 'lastfm':
        return dataloader.LastFM()
    else:
        raise ValueError(f'Dataset {config.dataset} not supported!')


def print_config_info(config):
    print('===========config================')
    pprint('To print our config')
    print("cores for test:", config.cores)
    print("comment:", config.comment)
    print("tensorboard:", config.tensorboard)
    print("LOAD:", config.load_bool)
    print("Weight path:", config.weight_path)
    print("Test Topks:", config.topks)
    print("using bpr loss")
    print('===========end===================')
