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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):

    def __init__(self, log_dir, log_name="log.txt"):
        self.logfile = open(os.path.join(log_dir, log_name), "w+")
    
    def log(self, msg):
        print(msg)
        print(msg, file=self.logfile)