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

from data.TCGAGBMDataset import ToTensor
from data.dataset import CrossValDataset
from data.samplers import CaseSampler
from models.cae import CAE, TestCAE
from models.mil import MultipleInstanceLearningClassifier
from models.factory import get_model
from utils.logging import make_exp_dir, save_checkpoint, AverageMeter, Logger, save_metrics, load_metrics


def get_targets(labels, case_ids, cases_order):
    """Get the right label for each case in cases_order."""
    targets = torch.zeros(len(cases_order))
    cases_ids = list(case_ids)
    for i, case in enumerate(cases_order):
        targets[i] = labels[cases_ids.index(case)]
    
    return targets


def get_data_loaders(config, device):
    dataset = CrossValDataset(
        config["dataset_name"],
        config["data_csv"], 
        config["data_dir"], 
        image_transform=torchvision.transforms.RandomCrop(config["input_size"]),
        sample_transform=ToTensor(device)
    )
    train_dataset = dataset.get_train_set()
    val_dataset = dataset.get_val_set()

    if config["dataset_name"] == "TCGAGBMDataset":
        train_dataset.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(config["input_size"]),
            torchvision.transforms.RandomRotation(config["rotation_angle"])
        ])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True,
            num_workers=4)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=config["eval_batch_size"], 
            sampler=torch.utils.data.RandomSampler(dataset.get_val_set(), replacement=True,
                                                num_samples=config["num_eval_samples"]), 
            num_workers=4)
    elif config["dataset_name"] == "PatchWiseDataset":
        cases_per_batch = 2
        patches_ber_case = 16
        train_sampler = CaseSampler(
            train_dataset, train_dataset.slides_frame, cases_per_batch, patches_ber_case
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler, batch_size=train_sampler.batch_size, num_workers=4
        )

        eval_cases_per_batch = 4
        eval_patches_per_case = 32
        val_sampler = CaseSampler(
            val_dataset, 
            val_dataset.slides_frame, 
            eval_cases_per_batch, 
            eval_patches_per_case,
            num_samples=config["num_eval_samples"]
        )
        val_loader = torch.utils.data.DataLoader(
            dataset.get_val_set(), 
            batch_size=val_sampler.batch_size, 
            sampler=val_sampler, 
            num_workers=4)
    
    return train_loader, val_loader


def compute_loss(config, model, batch, device):
    criterion = torch.nn.BCELoss()
    if config["model_name"] in ["cae", "test_cae"]:
        batch["label"] = batch["label"].to(device)
        loss = model.loss(batch["slide"], batch["label"])
    elif config["model_name"] == "mil_classifier":
        y_prob, cases = model(batch["slide"], batch["case_id"])
        target = get_targets(batch["label"], batch["case_id"], cases)
        loss = criterion(y_prob, target.to(device=device))
    
    return loss

def test(config, model, device, test_loader):
    model.eval()
    test_loss = 0
    batches = 0
    with torch.no_grad():
        for batch in test_loader:
            batch["slide"] = batch["slide"].to(device)
            batch["label"] = batch["label"].to(device)
            test_loss += compute_loss(config, model, batch, device).item()
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

    train_loader, val_loader = get_data_loaders(config, device)

    model = get_model(config["model_name"], **config["model_args"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

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

    # TODO: Track accuracy.
    batch_time = AverageMeter()
    train_losses = AverageMeter()
    val_losses = AverageMeter()

    keep_training = True
    while keep_training:
        for batch in train_loader:
            start = time.time()
            batch["slide"] = batch["slide"].to(device)

            model.train()
            optimizer.zero_grad()

            loss = compute_loss(config, model, batch, device)

            train_losses.update(loss.item())
            metrics["train_loss"].append(train_losses.val)

            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - start)

            if i_episode % config["eval_steps"] == 0:
                val_loss = test(config, model, device, val_loader)
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
                    "model_args": config["model_args"],
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
        device = "cuda" if config["use_gpu"] else "cpu"
        checkpoint = torch.load(config["resume"], map_location=device)
        exp_dir = os.path.dirname(config["resume"])
    else:
        checkpoint = None
        exp_dir = make_exp_dir(config["logging_dir"], config["exp_name"])

    with open(os.path.join(exp_dir, "config.yaml"), "w+") as f:
        yaml.dump(config, f, default_flow_style=False)

    main(config, exp_dir, checkpoint=checkpoint)