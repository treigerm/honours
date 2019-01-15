import os
import torch
import argparse
import pickle
import itertools
import pandas as pd
import numpy as np
import torchvision
from PIL import Image
from collections import defaultdict

from models.cae import CAE, TestCAE
from models.factory import get_model
from data.TCGAGBMDataset import TCGAGBMDataset, ToTensor
from data.dataset import CrossValDataset

def split_file_name(split, file_path):
    dirname = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    return os.path.join(dirname, "{}_{}".format(split, filename))

def main(model_name, checkpoint_path, root_dir, data_csv, batch_size, 
         out_file, use_gpu, num_load_workers):
    device = torch.device("cuda" if use_gpu else "cpu")
    model = get_model(model_name).to(device).eval()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    dataset = CrossValDataset(
        data_csv, root_dir, transform=ToTensor(device)
    )

    data_splits = {
        "train": dataset.get_train_set(),
        "val": dataset.get_val_set(),
        "test": dataset.get_test_set()
    }

    for name, data_split in data_splits.items():
        train_loader = torch.utils.data.DataLoader(
            data_split, batch_size=batch_size, shuffle=False, 
            num_workers=num_load_workers
        )

        embeddings = defaultdict(list)
        for batch in train_loader:
            slides = batch["slide"].to(device)

            embedding = model(slides)
            embedding = embedding.numpy()

            for i in range(len(batch)):
                case_id = slides["case_id"][i]
                embeddings[case_id].append(embedding[i])
        
        out_file_name = split_file_name(name, out_file)
        with open(out_file_name, "w+") as f:
            pickle.dump(embeddings, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--root-dir", type=str)
    parser.add_argument("--data-csv", type=str)
    parser.add_argument("--batch-size", type=str)
    parser.add_argument("--out-file", type=str)
    parser.add_argument("--use-gpu", type=bool)
    parser.add_argument("--num-load-workers", type=int, default=4)
    args = parser.parse_args()
    main(**vars(args))