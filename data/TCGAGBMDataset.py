import os
import pandas as pd # TODO: Replace with csv.
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class TCGAGBMDataset(Dataset):
    """TCGA GBM dataset.

    Read CSV file of the form (relative_path_to_image, label, patientID).
    """
    # TODO: Describe folder structure.

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.slides_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.slides_frame)
    
    def __getitem__(self, idx):
        # TODO: Have torch.Tensor of shape 3x128x128
        relative_path = self.slides_frame.iloc[idx, 0]
        slide_path = os.path.join(self.root_dir, relative_path)
        slide = np.array(Image.open(slide_path).convert("RGB"))
        label = self.slides_frame.iloc[idx, 1]
        sample = {"slide": slide, "label": label}

        if self.transform:
            sample = self.transform(sample)
        
        return sample