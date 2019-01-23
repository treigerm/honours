#!/usr/bin/env python

import os
import torch
import torchvision
import tensorboardX
import argparse
import yaml

from data.TCGAGBMDataset import TCGAGBMDataset, ToTensor
from data.dataset import CrossValDataset
from models.cae import CAE, TestCAE
from models.factory import get_model
from utils.logging import make_exp_dir, save_checkpoint


def test(model, device, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            slides = batch["slide"].to(device)
            test_loss += loss_fn(model(slides), slides)
    
    return test_loss / len(test_loader)


def main(config, exp_dir):
    torch.manual_seed(config["random_seed"])

    device = torch.device("cuda" if config["use_gpu"] else "cpu")

    writer = tensorboardX.SummaryWriter(os.path.join(config["tensorboard_dir"], 
                                                     os.path.basename(exp_dir)))

    dataset = CrossValDataset(
        config["data_csv"], 
        config["data_dir"], 
        image_transform=torchvision.transforms.RandomCrop(config["input_size"]),
        sample_transform=ToTensor(device)
    )
    train_dataset = dataset.get_train_set()
    train_dataset.set_image_transform(torchvision.transforms.Compose([
         torchvision.transforms.RandomCrop(config["input_size"]),
         torchvision.transforms.RandomRotation(config["rotation_angle"])
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        dataset.get_val_set(), batch_size=config["test_batch_size"], 
        shuffle=True, num_workers=4)

    model = get_model(config["model_name"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    i_episode = 1
    keep_training = True
    while keep_training:
        for batch in train_loader:
            slides = batch["slide"].to(device)

            model.train()
            optimizer.zero_grad()

            loss = loss_fn(model(slides), slides)
            loss.backward()
            optimizer.step()

            if i_episode % config["eval_steps"] == 0:
                val_loss = test(model, device, loss_fn, val_loader)

                print("Episode {}\ttrain loss: {:.4e} val loss: {:.4e}".format(
                    i_episode, loss, val_loss
                ))
                writer.add_scalars("data/losses", {"val_loss": val_loss,
                                                   "train_loss": loss}, i_episode)
                writer.export_scalars_to_json(os.path.join(exp_dir, "metrics.json"))

                is_best = val_loss < best_val_loss
                best_val_loss = val_loss if is_best else best_val_loss
                save_checkpoint({
                    "epoch": i_episode,
                    "model_name": config["model_name"],
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict()
                }, is_best, path=exp_dir)

            if i_episode == config["num_episodes"]:
                keep_training = False
                break

            i_episode += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, 
                        help="YAML file with configuration parameters")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f)
        for k, v in config.items():
            print("{}: {}".format(k, v))

    exp_dir = make_exp_dir(config["logging_dir"], config["exp_name"])

    with open(os.path.join(exp_dir, "config.yaml"), "w+") as f:
        yaml.dump(config, f, default_flow_style=False)

    main(config, exp_dir)