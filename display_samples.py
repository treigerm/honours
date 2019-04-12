#!/usr/bin/env python
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import argparse
import torchvision
import torch
import numpy as np
import random

from models.cae import CAE, TestCAE
from models.factory import get_model
from data.TCGAGBMDataset import TCGAGBMDataset, ToTensor
from data.dataset import CrossValDataset
from utils.logging import load_checkpoint

RANDOM_SEED = 42

def save_images(images, out_file=None, have_reconstructed=False):
    fig = plt.figure()

    if have_reconstructed:
        cols = len(images) / 2
        rows = 2
    else:
        cols = len(images)
        rows = 1
    for i in range(1, int(cols)+1):
        images_ix = 2 * (i - 1)
        ax = plt.subplot(rows, cols, i)
        ax.axis("off")
        plt.imshow(images[images_ix])
        if have_reconstructed:
            ax = plt.subplot(rows, cols, i + cols)
            ax.axis("off")
            plt.imshow(images[images_ix+1])

    plt.subplots_adjust(bottom=0.3, top=0.7)
    #plt.tight_layout()
    #plt.show()
    plt.savefig(out_file)


def main(root_dir, data_csv, crop_size, batch_size, checkpoint_path, use_gpu, 
         out_file):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    reconstruct_images = checkpoint_path is not None
    if reconstruct_images:
        device = torch.device("cuda" if use_gpu else "cpu")
        model, _ = load_checkpoint(checkpoint_path, device, get_model)
        model.eval()

    dataset = CrossValDataset(
        "PatchWiseDataset", data_csv, root_dir,
        image_transform=torchvision.transforms.RandomCrop(crop_size),
        sample_transform=ToTensor(None),
        rotate=False,
        flip=False,
        enhance=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset.get_train_set(), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for sample in data_loader:
            images = []
            if reconstruct_images:
                im_ae = model(sample["slide"].to(device))
                for i in range(len(sample["slide"])):
                    img = sample["slide"][i]
                    images.append(torchvision.transforms.functional.to_pil_image(img))
                    images.append(torchvision.transforms.functional.to_pil_image(im_ae[i]))
                save_images(images, out_file, have_reconstructed=True)
            else:
                for i in range(len(sample["slide"])):
                    img = sample["slide"][i]
                    images.append(torchvision.transforms.functional.to_pil_image(img))
                save_images(images, out_file, have_reconstructed=False)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--root-dir", type=str)
    parser.add_argument("--data-csv", type=str)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--out-file", default="samples.pdf")
    args = parser.parse_args()
    main(**vars(args))