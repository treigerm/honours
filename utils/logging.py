import os
import pickle
import datetime
import torch
import shutil

def save_metrics(metrics, exp_dir):
    out_file = os.path.join(exp_dir, "metrics.pickle")
    with open(out_file, "wb+") as f:
        metrics = {k: meter.values for k, meter in metrics.items()}
        pickle.dump(metrics, f)

def load_metrics(exp_dir):
    in_file = os.path.join(exp_dir, "metrics.pickle")
    with open(in_file, "rb") as f:
        metrics = pickle.load(f)
        return {k: AverageMeter(v) for k, v in metrics.items()}

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

def load_checkpoint(checkpoint_path, device, get_model):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(checkpoint["model_name"], **checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, values=None):
        self.reset()
        if values is not None:
            for v in values:
                self.update(v)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values.append(val)

class Logger(object):

    def __init__(self, log_dir, log_name="log.txt"):
        self.logfile = open(os.path.join(log_dir, log_name), "a")
    
    def log(self, msg):
        print(msg)
        print(msg, file=self.logfile)
        self.logfile.flush()