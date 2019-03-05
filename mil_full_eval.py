#!/usr/bin/env python

import argparse
import torch
import numpy as np
import random
import tqdm
import pickle

from data.TCGAGBMDataset import ToTensor
from data.dataset import CrossValDataset
from data.samplers import CaseUniqueSampler
from models.mil import MultipleInstanceLearningClassifier
from models.factory import get_model
from utils.logging import load_checkpoint

RANDOM_SEED = 42
CASES_PER_BATCH = 1

def get_data_loader(data_csv, root_dir, patches_per_case, device):
    dataset = CrossValDataset(
        "PatchWiseDataset",
        data_csv, 
        root_dir, 
        sample_transform=ToTensor(device)
    )
    val_dataset = dataset.get_val_set()

    val_sampler = CaseUniqueSampler(
        val_dataset,
        val_dataset.slides_frame,
        patches_per_case
    )
    return torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=val_sampler.batch_size, 
        sampler=val_sampler,
        num_workers=4
    )
    
def main(data_csv, root_dir, batches_per_case, patches_per_case, 
         checkpoint_path, use_gpu, out_file):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    device = torch.device("cuda" if use_gpu else "cpu")

    model, _ = load_checkpoint(checkpoint_path, device, get_model)
    data_loader = get_data_loader(data_csv, root_dir, patches_per_case, device)

    num_cases = len(data_loader)
    y_probs = np.zeros((num_cases, batches_per_case))
    y_true = np.zeros((num_cases))
    for i_batch in range(batches_per_case):
        for i_case, case_batch in tqdm.tqdm(enumerate(data_loader)):
            case_batch["slide"] = case_batch["slide"].to(device)
            y_prob, _ = model(
                case_batch["slide"], case_batch["case_id"])
            y_probs[i_case, i_batch] = y_prob
            y_true[i_case] = case_batch["label"][0]
    
    y_probs_means = np.mean(y_probs, axis=1)
    y_preds = y_probs_means > 0.5
    acc = np.sum(y_preds == y_true) / num_cases

    print()
    print("Final results:")
    print("Accuracy: {:.2f} %".format(acc * 100))
    with open(out_file, "wb") as f:
        pickle.dump({
            "Accuracy": acc,
            "y_probs": y_probs,
            "y_true": y_true
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv")
    parser.add_argument("--root-dir")
    parser.add_argument("--batches-per-case", type=int)
    parser.add_argument("--patches-per-case", type=int)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--out-file")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()
    main(**vars(args))