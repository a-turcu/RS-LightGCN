from world import Config
import dataloader
import model
import utils
from pprint import pprint


config = Config()

def load_dataset(dataset_str):
    if dataset_str in ['gowalla', 'yelp2018', 'amazon-book']:
        return dataloader.Loader(config)
    elif dataset_str == 'lastfm':
        return dataloader.LastFM()
    else:
        raise ValueError(f'Dataset {dataset_str} not supported!')

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
