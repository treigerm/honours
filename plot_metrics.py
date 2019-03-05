#!/usr/bin/env python

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pickle
import numpy as np
import argparse

from scipy.interpolate import make_interp_spline, BSpline

def make_plot(train_vals, val_vals, metric=None, step_size=100, title="metrics"):
    x_val = range(0,len(val_vals)*step_size, step_size)

    x_train = range(len(train_vals))
    x_train_smoothed = np.linspace(start=0, stop=len(train_vals), num=int(len(train_vals) * 0.1))
    spl = make_interp_spline(x_train, train_vals, k=1)
    y_train_smoothed = spl(x_train_smoothed)

    fig, ax = plt.subplots()
    ax.plot(train_vals, alpha=0.2, color="C0")
    ax.plot(x_train_smoothed, y_train_smoothed, label="Train", color="C0")
    ax.plot(x_val, val_vals, label="Validation", color="C1")
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric)
    ax.legend()
    plt.savefig("{}.pdf".format(title))

def main(metrics_file):
    with open(metrics_file, "rb") as f:
        metrics = pickle.load(f)

    make_plot(np.array(metrics["train_losses"]), np.array(metrics["val_losses"]),
              metric="Loss",
              title="train_losses")
    make_plot(np.array(metrics["train_accs"]), np.array(metrics["val_accs"]),
              metric="Accuracy",
              title="train_accs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file")
    args = parser.parse_args()
    main(**vars(args))