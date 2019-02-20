import os
import torch
import numpy as np

from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .factory import register_dataset

IMAGE_SIZE = (1000, 1000)
PATCH_SIZE = 128
STRIDE = PATCH_SIZE

@register_dataset("PatchWiseDataset")
class PatchWiseDataset(Dataset):
    """
    Based on: https://github.com/ImagingLab/ICIAR2018/blob/master/src/datasets.py
    """

    def __init__(self, data_frame, root_dir, 
                image_transform=None, 
                sample_transform=None,
                rotate=True,
                flip=True,
                enhance=True):
        self.slides_frame = data_frame
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.sample_transform = sample_transform

        wp = int((IMAGE_SIZE[0] - PATCH_SIZE) / STRIDE + 1)
        hp = int((IMAGE_SIZE[1] - PATCH_SIZE) / STRIDE + 1)

        self.shape = (len(self.slides_frame), wp, hp, 
            (4 if rotate else 1), (2 if flip else 1), (2 if enhance else 1))
    
    def __getitem__(self, index):
        idx, xpatch, ypatch, rotation, flip, enhance = np.unravel_index(index, self.shape)

        relative_path = self.slides_frame.iloc[idx, 0]
        slide_path = os.path.join(self.root_dir, relative_path)
        slide = Image.open(slide_path).convert("RGB")
        patch = slide.crop((
            xpatch * STRIDE, # left
            ypatch * STRIDE, # up
            xpatch * STRIDE + PATCH_SIZE, # right
            ypatch * STRIDE + PATCH_SIZE # down
        ))

        if rotation != 0:
            patch = patch.rotate(rotation * 90)

        if flip != 0:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)

        if enhance != 0:
            factors = np.random.uniform(.5, 1.5, 3)
            patch = ImageEnhance.Color(patch).enhance(factors[0])
            patch = ImageEnhance.Contrast(patch).enhance(factors[1])
            patch = ImageEnhance.Brightness(patch).enhance(factors[2])
        
        sample = {
            "slide": patch,
            "label": self.slides_frame.iloc[idx, 1],
            "case_id": self.slides_frame.iloc[idx, 2],
            "relative_path": relative_path
        }

        if self.sample_transform:
            sample = self.sample_transform(sample)
        
        return sample
    
    def __len__(self):
        return np.prod(self.shape)