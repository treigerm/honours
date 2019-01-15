import os
import pandas as pd

from .TCGAGBMDataset import TCGAGBMDataset

RANDOM_SEED = 42

class CrossValDataset(object):

    def __init__(self, csv_file, root_dir,  transform=None):
        train_frame, val_frame, test_frame = self.get_frames(csv_file)

        self.train_split = TCGAGBMDataset(train_frame, root_dir, transform)
        self.val_split = TCGAGBMDataset(val_frame, root_dir, transform)
        self.test_split = TCGAGBMDataset(test_frame, root_dir, transform)
    
    def get_frames(self, csv_file):
        csv_dir = os.path.dirname(csv_file)
        csv_name = os.path.basename(csv_dir)
        frames = []
        for split in ["train", "val", "test"]:
            file_name = os.path.join(csv_dir, "{}_{}".format(split, csv_name))
            frames.append(pd.read_csv(file_name))

        return frames
    
    def get_train_set(self):
        return self.train_split
    
    def get_val_set(self):
        return self.val_split
    
    def get_test_set(self):
        return self.test_split