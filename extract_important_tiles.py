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
from models.factory import get_model
from data.TCGAGBMDataset import TCGAGBMDataset, ToTensor
from data.dataset import CrossValDataset

RANDOM_SEED = 42
INPUT_SIZE = 128

def update_top_images(slides, scores, top, k=10):
    top_ixs = np.argpartition(scores, k)[-k:]
    top_scores = zip(top_ixs, scores[top_ixs])
    for ix, score in top_scores:
        if len(top) < k or score > top[0]:
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

def main(importances_file, use_gpu, checkpoint_path, data_csv, root_dir, 
         batch_size, num_load_workers, out_file):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    with open(importances_file, "rb") as f:
        coef = pickle.load(f)
        feature_importances = coef["feature_importances"]

    device = torch.device("cuda" if use_gpu else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(checkpoint["model_name"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])

    dataset = CrossValDataset(
        data_csv, root_dir, 
        image_transform=torchvision.transforms.RandomCrop(INPUT_SIZE),
        sample_transform=ToTensor(device)
    )

    best_dim = np.argmax(feature_importances)

    data_loader = torch.utils.data.DataLoader(
        dataset.get_train_set(), batch_size=batch_size, shuffle=False,
        num_workers=num_load_workers
    )

    top_ten_images = []

    with torch.no_grad():
        for batch in data_loader:
            slides = batch["slide"]
            slides_gpu = slides.to(device)

            embeddings = model.encoder(slides_gpu)
            emb_best_dim = embeddings[:, best_dim].cpu().numpy()

            top_ten_images = update_top_images(slides, emb_best_dim, top_ten_images)

    images = []
    for score, image in top_ten_images:
        images.append(torchvision.transforms.functional.to_pil_image(image))
    
    save_images(images, 10, out_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))