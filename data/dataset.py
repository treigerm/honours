import torch
import numpy as np
import pandas as pd

from .TCGAGBMDataset import TCGAGBMDataset

RANDOM_SEED = 42

class CrossValDataset(object):
    # TODO: Make this more PyTorch idiomatic i.e. use torch.utils.data.random_split()

    def __init__(self, csv_file, root_dir, train_size, 
                 val_size, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        train_idxs, val_idxs, test_idxs = self.split_dataset(train_size, 
                                                             val_size)

        self.train_split = TCGAGBMDataset(self.data_frame.iloc[train_idxs], 
                                          root_dir, transform)
        self.val_split = TCGAGBMDataset(self.data_frame.iloc[val_idxs],
                                        root_dir, transform)
        self.test_split = TCGAGBMDataset(self.data_frame.iloc[test_idxs],
                                         root_dir, transform)
    
    def split_dataset(self, train_size, val_size):
        rnd = np.random.RandomState(RANDOM_SEED)
        num_elements = len(self.data_frame)
        ixs = np.arange(num_elements)
        rnd.shuffle(ixs)
        num_train = int(np.floor(num_elements * train_size))
        num_val = int(np.floor(num_elements * val_size))
        return np.split(ixs, [num_train, num_train+num_val])

    
    def get_train_set(self):
        return self.train_split
    
    def get_val_set(self):
        return self.val_split
    
    def get_test_set(self):
        return self.test_split