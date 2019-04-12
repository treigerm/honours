#!/usr/bin/env python

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pickle
import numpy as np
import argparse

from scipy.interpolate import make_interp_spline, BSpline

matplotlib.rcParams.update({'font.size': 22})


def make_plot(train_vals, val_vals, metric=None, step_size=100, title="metrics", 
              legend=True):
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
    if legend:
        ax.legend()
    fig.tight_layout()
    plt.savefig("{}.pdf".format(title))

def main(metrics_file, no_legend_acc, no_legend_loss):
    with open(metrics_file, "rb") as f:
        metrics = pickle.load(f)

    make_plot(np.array(metrics["train_losses"]), np.array(metrics["val_losses"]),
              metric="Loss",
              title="train_losses",
              legend=(not no_legend_loss))
    make_plot(np.array(metrics["train_accs"]), np.array(metrics["val_accs"]),
              metric="Accuracy",
              title="train_accs",
              legend=(not no_legend_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file")
    parser.add_argument("--no-legend-acc", action="store_true")
    parser.add_argument("--no-legend-loss", action="store_true")
    args = parser.parse_args()
    main(**vars(args))