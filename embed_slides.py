#!/usr/bin/env python
import os
import torch
import torchvision
import argparse
import pickle
import tqdm
import numpy as np
import random
from collections import defaultdict

from models.cae import CAE, TestCAE
from models.factory import get_model
from data.TCGAGBMDataset import TCGAGBMDataset, ToTensor
from data.samplers import CaseSampler
from data.dataset import CrossValDataset
from utils.logging import load_checkpoint

RANDOM_SEED = 42
INPUT_SIZE = 128

PATCHES_PER_CASE = 64
CASES_PER_BATCH = 8

def aggregate_embeddings(embeddings):
    """
    Args:
        embeddings (np.array): size of (num_embeddings, num_hidden_dims)
    """
    mean = np.mean(embeddings, axis=0)
    median = np.median(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    return {"mean": mean, "std": std, "median": median}

def main(checkpoint_path, root_dir, data_csv, batch_size, 
         num_samples, out_file, use_gpu, num_load_workers, aggregate):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    device = torch.device("cuda" if use_gpu else "cpu")
    model, _ = load_checkpoint(checkpoint_path, device, get_model)
    model.eval()

    dataset = CrossValDataset(
        "PatchWiseDataset", data_csv, root_dir, 
        sample_transform=ToTensor(device),
        rotate=False,
        flip=False,
        enhance=False
    )

    data_splits = {
        "train": dataset.get_train_set(),
        "val": dataset.get_val_set(),
        "test": dataset.get_test_set()
    }

    results = {}
    out_dir = os.path.dirname(out_file)
    out_filename = os.path.basename(out_file)
    for name, data_split in data_splits.items():
        #data_loader = torch.utils.data.DataLoader(
        #    data_split, batch_size=batch_size, shuffle=False, 
        #    num_workers=num_load_workers
        #)
        sampler = CaseSampler(
            data_split,
            data_split.slides_frame,
            CASES_PER_BATCH,
            PATCHES_PER_CASE
        )
        data_loader = torch.utils.data.DataLoader(
            data_split, 
            batch_size=sampler.batch_size, 
            sampler=sampler,
            num_workers=num_load_workers
        )

        embeddings = defaultdict(list)
        with torch.no_grad():
            with tqdm.tqdm(total=len(data_loader)) as pbar:
                pbar.set_description("{}".format(name))
                for batch in data_loader:
                    slides = batch["slide"].to(device)

                    embedding = model.encoder(slides)
                    embedding = embedding.cpu().numpy()

                    for i in range(len(batch["case_id"])):
                        case_id = batch["case_id"][i]
                        relative_path = batch["relative_path"][i]
                        embeddings[case_id].append(
                            (relative_path, embedding[i])
                        )
                    pbar.set_description("Patients {}".format(len(embeddings)))
                    pbar.update(1)
        
        if aggregate:
            for case_id, case_embeddings in embeddings.items():
                # case_embeddings: [(relative_path, embeddings)]
                case_embeddings = np.array([x[1] for x in case_embeddings])
                embeddings[case_id] = aggregate_embeddings(case_embeddings)
        out = os.path.join(out_dir, "{}_{}".format(name, out_filename))
        with open(out, "wb+") as f:
            pickle.dump(embeddings, f, protocol=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="model_best.pth.tar")
    parser.add_argument("--root-dir", type=str, 
        default="/exports/igmm/eddie/batada-lab/timreichlet")
    parser.add_argument("--data-csv", type=str, 
        default="/exports/igmm/eddie/batada-lab/timreichlet/metadata/all_non_background/metadata_top.csv")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=10,
        help="Number of samples from each 1000x1000 tile. Not relevant for PatchWiseDataset.")
    parser.add_argument("--out-file", type=str, default="embeddings.pickle")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--num-load-workers", type=int, default=4)
    args = parser.parse_args()
    main(**vars(args))
