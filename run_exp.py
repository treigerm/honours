#!/usr/bin/env python

import torch
from data.TCGAGBMDataset import TCGAGBMDataset

# TODO: Save config.
# TODO: Training, validation, test split.
# TODO: Implement autoencoder model.
# TODO: Implement GPU support.

DATA_DIR = "/Users/Tim/data/tcga_luad/gdc_download_20181013_213421.982108"
DATA_CSV = "/Users/Tim/data/tcga_luad/gdc_download_20181013_213421.982108/tile_locations.csv"

def main(config):
    dataset = TCGAGBMDataset(DATA_CSV, DATA_DIR)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    for i_batch, batch in enumerate(dataloader):
        print(batch["slide"].size())
        print(batch["label"].size())
        print(batch["label"])
        break

if __name__ == "__main__":
    config = {
        "batch_size": 4
    }
    main(config)