#!/usr/bin/env python

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import torch
import torchvision
import numpy as np
import argparse
import heapq
import pickle
import random

from models.cae import CAE, TestCAE
from models.mil import MultipleInstanceLearningClassifier
from models.factory import get_model
from data.TCGAGBMDataset import TCGAGBMDataset, ToTensor
from data.dataset import CrossValDataset
from data.samplers import CaseSampler
from utils.logging import load_checkpoint

RANDOM_SEED = 42
INPUT_SIZE = 128

CASES_PER_PATCH = 4
PATCHES_PER_CASE = 64

def update_top_images(slides, scores, top, k=10):
    top_k = min(k, len(scores))
    top_ixs = np.argpartition(scores, top_k)[-top_k:]
    top_scores = zip(top_ixs, scores[top_ixs])
    for ix, score in top_scores:
        if len(top) < k or score > top[0][0]:
            if len(top) == k:
                heapq.heappop(top)
            
            heapq.heappush(top, (score, slides[ix]))
    
    return top

def save_images(images, k, out_file=None):
    fig = plt.figure()

    for i in range(len(images)):
        ax = plt.subplot(k, 1, i + 1)
        #plt.tight_layout()
        #ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(images[i])
    
    plt.savefig(out_file)

def main(use_gpu, checkpoint_path, data_csv, root_dir, 
         batch_size, num_load_workers, out_file, num_samples):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    device = torch.device("cuda" if use_gpu else "cpu")
    model, _ = load_checkpoint(checkpoint_path, device, get_model)
    model.eval()

    dataset = CrossValDataset(
        "PatchWiseDataset", data_csv, root_dir, 
        rotate=False, flip=False, enhance=False,
        sample_transform=ToTensor(device)
    )

    train_dataset = dataset.get_train_set()
    train_sampler = CaseSampler(
        train_dataset, 
        train_dataset.slides_frame, 
        CASES_PER_PATCH, 
        PATCHES_PER_CASE,
        num_samples=num_samples
    )
    data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=train_sampler.batch_size, num_workers=4
    )

    top_ten_images = []

    print("Embedding images..")
    with torch.no_grad():
        for batch in data_loader:
            slides = batch["slide"]
            slides_gpu = slides.to(device)

            embeddings = model.encoder(slides_gpu)
            att = model.attention(embeddings)
            scores = att.cpu().numpy().reshape((-1))

            top_ten_images = update_top_images(slides, scores, top_ten_images)

    images = []
    for score, image in top_ten_images:
        images.append(torchvision.transforms.functional.to_pil_image(image))
    
    save_images(images, 10, out_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--root-dir", type=str)
    parser.add_argument("--data-csv", type=str)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--out-file", default="att_tiles.png")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--num-load-workers", type=int, default=4)
    args = parser.parse_args()
    main(**vars(args))