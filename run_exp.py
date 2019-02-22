#!/usr/bin/env python

import os
import torch
import torchvision
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
from utils.logging import make_exp_dir, save_checkpoint, AverageMeter, Logger, save_metrics, load_metrics, load_checkpoint


def calculate_accuracy(y_prob, y_true):
    y_hat = torch.ge(y_prob, 0.5).float()
    return y_hat.eq(y_true).float().mean().cpu().item()


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
        train_sampler = CaseSampler(
            train_dataset, 
            train_dataset.slides_frame, 
            config["cases_per_batch"], 
            config["patches_per_case"]
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler, batch_size=train_sampler.batch_size, num_workers=4
        )

        val_sampler = CaseSampler(
            val_dataset, 
            val_dataset.slides_frame, 
            config["eval_cases_per_batch"], 
            config["eval_patches_per_case"],
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
        # TODO: Reshape y_prob.
        loss = criterion(y_prob, target.to(device))
        accuracy = calculate_accuracy(y_prob, target.to(device))
    
    return {
        "loss": loss,
        "accuracy": accuracy
    }

def test(config, model, device, test_loader):
    model.eval()
    test_loss = 0
    test_acc = 0
    batches = 0
    with torch.no_grad():
        for batch in test_loader:
            batch["slide"] = batch["slide"].to(device)
            batch["label"] = batch["label"].to(device)
            scores = compute_loss(config, model, batch, device)
            test_loss += scores["loss"].item()
            test_acc += scores["accuracy"]
            batches += 1
    
    return test_loss / batches, test_acc / batches


def main(config, exp_dir, checkpoint=None):
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    logger = Logger(exp_dir)

    device = torch.device("cuda" if config["use_gpu"] else "cpu")

    train_loader, val_loader = get_data_loaders(config, device)

    model = get_model(config["model_name"], **config["model_args"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    if "load_encoder" in config:
        encoder_model, _ = load_checkpoint(
            config["load_encoder"], device, get_model)
        model.encoder = encoder_model.encoder

    if checkpoint:
        logger.log("Resume training..")
        metrics = load_metrics(exp_dir)
        best_val_loss = checkpoint["best_val_loss"]
        i_episode = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    else: 
        i_episode = 1
        best_val_loss = float("inf")
        metrics = {
            "batch_time": AverageMeter(),
            "val_time": AverageMeter(),
            "train_losses": AverageMeter(),
            "val_losses": AverageMeter(),
            "train_accs": AverageMeter(),
            "val_accs": AverageMeter()
        }

    keep_training = True
    while keep_training:
        for batch in train_loader:
            start = time.time()
            batch["slide"] = batch["slide"].to(device)

            model.train()
            optimizer.zero_grad()

            scores = compute_loss(config, model, batch, device)
            loss, acc = scores["loss"], scores["accuracy"]

            metrics["train_losses"].update(loss.item())
            metrics["train_accs"].update(acc)

            loss.backward()
            optimizer.step()
            metrics["batch_time"].update(time.time() - start)

            if i_episode % config["eval_steps"] == 0:
                val_start = time.time()
                val_loss, val_acc = test(config, model, device, val_loader)
                scheduler.step(val_loss)
                metrics["val_time"].update(time.time() - val_start)
                metrics["val_losses"].update(val_loss)
                metrics["val_accs"].update(val_acc)

                # Our optimizer has only one parameter group so the first 
                # element of our list is our learning rate.
                lr = optimizer.param_groups[0]['lr']
                # TODO: Print validation time.
                logger.log(
                    "Episode {0}\t"
                    "Time {metrics[batch_time].val:.3f} ({metrics[batch_time].avg:.3f}) "
                    "Val time {metrics[val_time].val:.3f} ({metrics[val_time].avg:.3f}) "
                    "Train loss {metrics[train_losses].val:.4e} ({metrics[train_losses].avg:.4e}) "
                    "Train acc {metrics[train_accs].val:.4f} ({metrics[train_accs].avg:.4f}) "
                    "Val loss {metrics[val_losses].val:.4e} ({metrics[val_losses].avg:.4e}) "
                    "Val acc {metrics[val_accs].val:.4f} ({metrics[val_accs].avg:.4f}) "
                    "Learning rate {lr:.2e}".format(
                    i_episode, lr=lr, metrics=metrics)
                )

                save_metrics(metrics, exp_dir)

                is_best = val_loss < best_val_loss
                best_val_loss = val_loss if is_best else best_val_loss
                save_checkpoint({
                    "epoch": i_episode,
                    "model_name": config["model_name"],
                    "model_args": config["model_args"],
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
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