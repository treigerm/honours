import os
import pandas as pd # TODO: Replace with csv.
import itertools
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF

SLIDE_WIDTH = 1000
SLIDE_HEIGHT = 1000

ITEM_WIDTH = 128
ITEM_HEIGHT = 128

class TCGAGBMDataset(Dataset):
    """TCGA GBM dataset.

    Read CSV file of the form (relative_path_to_image, label, case_id).
    """
    # TODO: Describe folder structure.
    # TODO: Describe tiling structure.

    def __init__(self, data_frame, root_dir, transform=None):
        """
        Args:
            data_frame (pandas.DataFrame)
            root_dir (String)
        """
        self.slides_frame = data_frame
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
    
    def set_transform(self, transform):
        self.transform = transform

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
        case_id = self.slides_frame.iloc[slide_idx, 2] 
        sample = {
            "slide": slide, 
            "label": label, 
            "case_id": case_id,
            "relative_path": relative_path
        }

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    """Convert numpy.ndarray in sample to torch.Tensor."""

    def __init__(self, device):
        self.device = device
    
    def __call__(self, sample):
        slide, label = sample["slide"], sample["label"]

        # Transforms image data from shape (W, H, C) to (C, W, H) and from 
        # range (0, 255) to (0, 1).
        sample["slide"] = torchvision.transforms.functional.to_tensor(slide)
        sample["label"] = torch.tensor(label).float()
        return sample

class RandomRotate(object):

    def __init__(self, degrees):
        if isinstance(degrees, int):
            self.degrees = (-degrees, degrees)
        elif isinstance(degrees, (list, tuple)) and len(degrees) == 2:
            self.degrees = degrees

    def __call__(self, sample):
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        slide = Image.fromarray(sample["slide"])
        sample["slide"] = np.array(TF.rotate(slide, angle))
        return sample