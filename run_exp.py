#!/usr/bin/env python

import torch
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

TRAIN_SIZE = 0.7
VAL_SIZE = 0.2

def test(model, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            slides = batch["slide"]
            test_loss += loss_fn(model(slides), slides)
            num_batches += 1
    
    test_loss /= num_batches
    print("Average loss: {:.4f}".format(test_loss))


def main(config):
    device = DEVICE
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
        for i_batch, batch in enumerate(train_loader):
            slides = batch["slide"]
            print(slides[0, :, 0, 0])

            model.train()
            optimizer.zero_grad()

            loss = loss_fn(model(slides), slides)
            loss.backward()
            optimizer.step()
            if i_batch == 4:
                break
        
        print("Epoch {}".format(i_epoch))
        test(model, loss_fn, val_loader)

if __name__ == "__main__":
    config = {
        "batch_size": 4,
        "test_batch_size": 100,
        "learning_rate": 5e-3,
        "weight_decay": 1e-5,
        "num_epochs": 100
    }
    main(config)