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

INPUT_SIZE = 1000  # Input slides are 1000x1000
OUTPUT_SIZE = 128  # Output slides should be 128x128

def subslides_indexes():
    ixs = []
    current_ix = 0
    while True:
        if current_ix + OUTPUT_SIZE < INPUT_SIZE:
            ixs.append(current_ix)
            current_ix += OUTPUT_SIZE
        else:
            ixs.append(INPUT_SIZE - OUTPUT_SIZE)
            break
    return ixs

def make_embeddings(model, slide, device):
    embeddings = []
    ixs = itertools.product(subslides_indexes(), subslides_indexes())
    slide = slide.to(device)
    for x_sub_slide, y_sub_slide in ixs:
        embedding = model.encoder(
            slide[x_sub_slide:(x_sub_slide + OUTPUT_SIZE), 
                    y_sub_slide:(y_sub_slide + OUTPUT_SIZE)]
            )
        embeddings.append(embedding.numpy())

    return embeddings

def get_slide_id(slide_path):
    return os.path.basename(os.path.dirname(slide_path))

def get_slide(root_dir, relative_path):
    slide_path = os.path.join(root_dir, relative_path)
    slide = np.array(Image.open(slide_path).convert("RGB"))

    # Transforms image data from shape (W, H, C) to (C, W, H) and from 
    # range (0, 255) to (0, 1).
    slide = torchvision.transforms.functional.to_tensor(slide)
    return slide

def main(model_name, checkpoint_path, root_dir, slides_csv, out_file, use_gpu):
    device = torch.device("cuda" if use_gpu else "cpu")
    model = get_model(model_name).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    slides = pd.read_csv(slides_csv)

    embeddings = defaultdict(list)
    for slide_path in slides["relative_path"]:
        slide_id = get_slide_id(slide_path)
        slide = get_slide(root_dir, slide_path)
        embeddings[slide_id].append(make_embeddings(model, slide, device))
    
    with open(out_file, "w+") as f:
        pickle.dump(embeddings, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--root-dir", type=str)
    parser.add_argument("--slides-csv", type=str)
    parser.add_argument("--out-file", type=str)
    parser.add_argument("--use-gpu", type=bool)
    args = parser.parse_args()
    main(**vars(args))