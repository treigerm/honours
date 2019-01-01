#!/usr/bin/env python

import torch
from data.TCGAGBMDataset import TCGAGBMDataset
from models.CAE import CAE

# TODO: Save config.
# TODO: Training, validation, test split.
# TODO: Implement GPU support.

DATA_DIR = "/Users/Tim/data/tcga_luad/gdc_download_20181013_213421.982108"
DATA_CSV = "/Users/Tim/data/tcga_luad/gdc_download_20181013_213421.982108/tile_locations.csv"

def main(config):
    dataset = TCGAGBMDataset(DATA_CSV, DATA_DIR)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    model = CAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    loss_fn = torch.nn.MSELoss()
    for i_batch, batch in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()

        loss = loss_fn(model(batch["slide"]), batch["slide"])
        loss.backward()
        optimizer.step()
        if i_batch == 4:
            break

    print("Test passed")

if __name__ == "__main__":
    config = {
        "batch_size": 4,
        "learning_rate": 5e-3,
        "weight_decay": 1e-5
    }
    main(config)