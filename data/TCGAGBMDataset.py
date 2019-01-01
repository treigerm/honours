import os
import pandas as pd # TODO: Replace with csv.
import itertools
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SLIDE_WIDTH = 1000
SLIDE_HEIGHT = 1000

ITEM_WIDTH = 128
ITEM_HEIGHT = 128

class TCGAGBMDataset(Dataset):
    """TCGA GBM dataset.

    Read CSV file of the form (relative_path_to_image, label, patientID).
    """
    # TODO: Describe folder structure.
    # TODO: Describe tiling structure.

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.slides_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.sub_slides = list(itertools.product(
            self.item_indexes(SLIDE_WIDTH, ITEM_WIDTH),
            self.item_indexes(SLIDE_HEIGHT, ITEM_HEIGHT)
        ))
    
    @staticmethod
    def item_indexes(item_length, step_size):
        # TODO: Put function into utility module.
        ixs = []
        current_ix = 0
        while True:
            if current_ix + step_size < item_length:
                ixs.append(current_ix)
                current_ix += step_size
            else:
                ixs.append(item_length - step_size)
                break
        return ixs

    def __len__(self):
        return len(self.slides_frame) * len(self.sub_slides)
    
    def __getitem__(self, idx):
        slide_idx = int(np.floor(idx / len(self.sub_slides)))

        sub_slide_idx = idx % len(self.sub_slides)
        x_sub_slide, y_sub_slide = self.sub_slides[sub_slide_idx]

        relative_path = self.slides_frame.iloc[slide_idx, 0]
        slide_path = os.path.join(self.root_dir, relative_path)
        slide = np.array(Image.open(slide_path).convert("RGB"))
        slide = slide[x_sub_slide:(x_sub_slide + ITEM_WIDTH), 
                      y_sub_slide:(y_sub_slide + ITEM_HEIGHT)]

        label = self.slides_frame.iloc[slide_idx, 1]
        # Change tensore from (width, height, channels) to 
        # (channels, width, height).
        slide = torch.tensor(slide, dtype=torch.float).permute(2, 0, 1)
        sample = {"slide": slide, "label": label}

        if self.transform:
            sample = self.transform(sample)
        
        return sample