import os
from world import Config
from dataloader import Loader


dataset_name = 'gowalla'
# Instantiate the config object
config = Config()
dataset = Loader(config, minimal_bool=True)

