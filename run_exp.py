#!/usr/bin/env python

import os
import torch
import torchvision
import tensorboardX
import argparse
import yaml
import pickle
import time
import numpy as np
import random

from data.TCGAGBMDataset import TCGAGBMDataset, ToTensor
from data.dataset import CrossValDataset
from models.cae import CAE, TestCAE
from models.factory import get_model
from utils.logging import make_exp_dir, save_checkpoint, AverageMeter, Logger, save_metrics, load_metrics

def test(model, device, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    batches = 0
    with torch.no_grad():
        for batch in test_loader:
            slides = batch["slide"].to(device)
            test_loss += loss_fn(model(slides), slides)
            batches += 1
    
    return test_loss / batches


def main(config, exp_dir, checkpoint=None):
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    logger = Logger(exp_dir)

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
        dataset.get_val_set(), 
        batch_size=config["eval_batch_size"], 
        sampler=torch.utils.data.RandomSampler(dataset.get_val_set(), replacement=True,
                                               num_samples=config["num_eval_samples"]), 
        num_workers=4)

    model = get_model(config["model_name"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    loss_fn = torch.nn.MSELoss()

    if checkpoint:
        logger.log("Resume training..")
        metrics = load_metrics(exp_dir)
        best_val_loss = checkpoint["best_val_loss"]
        i_episode = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else: 
        i_episode = 1
        metrics = {"val_loss": [], "train_loss": []}
        best_val_loss = float("inf")

    batch_time = AverageMeter()
    train_losses = AverageMeter()
    val_losses = AverageMeter()

    keep_training = True
    while keep_training:
        for batch in train_loader:
            start = time.time()
            slides = batch["slide"].to(device)

            model.train()
            optimizer.zero_grad()

            loss = loss_fn(model(slides), slides)
            train_losses.update(loss)
            metrics["train_loss"].append(train_losses.val)

            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - start)

            if i_episode % config["eval_steps"] == 0:
                val_loss = test(model, device, loss_fn, val_loader)
                scheduler.step(val_loss)
                val_losses.update(val_loss)

                # Our optimizer has only one parameter group so the first 
                # element of our list is our learning rate.
                lr = optimizer.param_groups[0]['lr']
                logger.log(
                    "Episode {0}\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    "Train loss {train_loss.val:.4e} ({train_loss.avg:.4e}) "
                    "Val loss {val_loss.val:.4e} ({val_loss.avg:.4e}) "
                    "Learning rate {lr:.2e}".format(
                    i_episode, val_loss=val_losses, batch_time=batch_time, 
                    train_loss=train_losses, lr=lr)
                )

                writer.add_scalars("data/losses", {"val_loss": val_loss,
                                                   "train_loss": train_losses.val}, i_episode)
                metrics["val_loss"].append(val_loss)
                save_metrics(metrics, exp_dir)

                is_best = val_loss < best_val_loss
                best_val_loss = val_loss if is_best else best_val_loss
                save_checkpoint({
                    "epoch": i_episode,
                    "model_name": config["model_name"],
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict()
                }, is_best, path=exp_dir)

            if i_episode >= config["num_episodes"]:
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


    if "resume" in config:
        checkpoint = torch.load(config["resume"])
        exp_dir = os.path.dirname(config["resume"])
    else:
        checkpoint = None
        exp_dir = make_exp_dir(config["logging_dir"], config["exp_name"])

    with open(os.path.join(exp_dir, "config.yaml"), "w+") as f:
        yaml.dump(config, f, default_flow_style=False)

    main(config, exp_dir, checkpoint=checkpoint)