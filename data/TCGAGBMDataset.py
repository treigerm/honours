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

    def __init__(self, data_frame, root_dir, 
                 image_transform=None,
                 sample_transform=None):
        """
        Args:
            data_frame (pandas.DataFrame)
            root_dir (String)
        """
        self.slides_frame = data_frame
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.sample_transform = sample_transform

    def set_image_transform(self, transform):
        self.image_transform = transform

    def __len__(self):
        return len(self.slides_frame)
    
    def __getitem__(self, idx):
        relative_path = self.slides_frame.iloc[idx, 0]
        slide_path = os.path.join(self.root_dir, relative_path)
        slide = Image.open(slide_path).convert("RGB")

        label = self.slides_frame.iloc[idx, 1]
        case_id = self.slides_frame.iloc[idx, 2] 

        if self.image_transform:
            slide = self.image_transform(slide)

        sample = {
            "slide": slide, 
            "label": label, 
            "case_id": case_id,
            "relative_path": relative_path
        }

        if self.sample_transform:
            sample = self.sample_transform(sample)
        
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
