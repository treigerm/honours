#!/usr/bin/env python

import torch
import os
import tensorboardX
import argparse
import yaml
import tqdm

from data.TCGAGBMDataset import TCGAGBMDataset, ToTensor
from data.dataset import CrossValDataset
from models.CAE import CAE

# TODO: Save config.
# TODO: Save best model.
# TODO: Implement logging.
# TODO: Set random seeds.

DEVICE = "cpu"

DATA_DIR = "/Users/Tim/data/tcga_luad/gdc_download_20181013_213421.982108"
DATA_CSV = "/Users/Tim/data/tcga_luad/gdc_download_20181013_213421.982108/tile_locations.csv"

TENSORBOARD_DIR = "/Users/Tim/dev/cw/honours/tensorboard"
LOGGING_DIR = "/Users/Tim/dev/cw/honours/logs"

TRAIN_SIZE = 0.7
VAL_SIZE = 0.2

def test(model, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            slides = batch["slide"]
            test_loss += loss_fn(model(slides), slides)
    
    return test_loss / len(test_loader)


def main(config):
    device = DEVICE

    writer = tensorboardX.SummaryWriter(os.path.join(TENSORBOARD_DIR, 
                                                     config["exp_name"]))

    dataset = CrossValDataset(DATA_CSV, DATA_DIR, TRAIN_SIZE, VAL_SIZE, 
                              transform=ToTensor(device))
    train_loader = torch.utils.data.DataLoader(
        dataset.get_train_set(), batch_size=config["batch_size"], shuffle=True,
        num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        dataset.get_val_set(), batch_size=config["test_batch_size"], 
        shuffle=True, num_workers=4)

    model = CAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    loss_fn = torch.nn.MSELoss()
    for i_epoch in range(config["num_epochs"]):
        train_loss = 0
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for i_batch, batch in enumerate(train_loader):
                slides = batch["slide"]

                model.train()
                optimizer.zero_grad()

                loss = loss_fn(model(slides), slides)
                writer.add_scalar("data/batch_loss", loss, i_batch)
                train_loss += loss
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_description("loss: {:.4f}".format(loss))

        print("Epoch {}".format(i_epoch))

        train_loss /= len(train_loader)
        test_loss = test(model, loss_fn, val_loader)
        writer.add_scalar("data/loss", {"train_loss": train_loss, 
                                        "test_loss": test_loss}, i_epoch)
        
        if i_epoch == 1:
            break
    
    writer.export_scalars_to_json(os.path.join(LOGGING_DIR, config["exp_name"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, 
                        help="YAML file with configuration parameters")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f)
        for k, v in config.items():
            print("{}: {}".format(k, v))

    main(config)