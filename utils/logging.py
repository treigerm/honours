import os
import datetime
import torch
import shutil

def make_exp_dir(log_dir, exp_name):
    exp_dir_name = "{}_{}".format(
        exp_name, 
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    exp_dir = os.path.join(log_dir, exp_dir_name)
    os.makedirs(exp_dir)
    return exp_dir


def save_checkpoint(state, is_best, path=".", filename="checkpoint.pth.tar"):
    filepath = os.path.join(path, filename)
    torch.save(state, filepath)
    if is_best:
        best_file = os.path.join(path, "model_best.pth.tar")
        shutil.copyfile(filepath, best_file)