#!/usr/bin/env python

import torch
import os
import tensorboardX
import argparse
import yaml
import tqdm

from data.TCGAGBMDataset import TCGAGBMDataset, ToTensor
from data.dataset import CrossValDataset
from models.cae import CAE, TestCAE
from models.factory import get_model
from utils.logging import make_exp_dir, save_checkpoint

RANDOM_SEED = 42

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


def main(config, exp_dir):
    torch.manual_seed(RANDOM_SEED)

    device = DEVICE

    writer = tensorboardX.SummaryWriter(os.path.join(TENSORBOARD_DIR, 
                                                     os.path.basename(exp_dir)))

    dataset = CrossValDataset(DATA_CSV, DATA_DIR, TRAIN_SIZE, VAL_SIZE, 
                              transform=ToTensor(device))
    train_loader = torch.utils.data.DataLoader(
        dataset.get_train_set(), batch_size=config["batch_size"], shuffle=True,
        num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        dataset.get_val_set(), batch_size=config["test_batch_size"], 
        shuffle=True, num_workers=4)

    model = get_model(config["model_name"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
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

        train_loss /= len(train_loader)
        val_loss = test(model, loss_fn, val_loader)

        print("Epoch {} train loss: {:.4f} val loss: {:.4f}".format(
            i_epoch + 1, train_loss, val_loss
        ))
        writer.add_scalars("data/loss", {"train_loss": train_loss, 
                                        "val_loss": val_loss}, i_epoch)

        is_best = val_loss < best_val_loss
        save_checkpoint({
            "epoch": i_epoch + 1,
            "model_name": config["model_name"],
            "state_dict": model.state_dict(),
            "best_val_loss": best_val_loss,
            "optimizer": optimizer.state_dict()
        }, is_best, path=exp_dir)
    
    writer.export_scalars_to_json(os.path.join(exp_dir, "metrics.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, 
                        help="YAML file with configuration parameters")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f)
        for k, v in config.items():
            print("{}: {}".format(k, v))

    exp_dir = make_exp_dir(LOGGING_DIR, config["exp_name"])

    with open(os.path.join(exp_dir, "config.yaml"), "w+") as f:
        yaml.dump(config, f, default_flow_style=False)

    main(config, exp_dir)