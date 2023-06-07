import os


os.getcwd()

os.listdir('code')

from dataloader import Loader

dataset_name = 'gowalla'


dataset = Loader(path="../data/" + dataset_name)

