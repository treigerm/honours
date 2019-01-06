import os
import datetime

def make_exp_dir(log_dir, exp_name):
    exp_dir_name = "{}_{}".format(
        exp_name, 
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    exp_dir = os.path.join(log_dir, exp_dir_name)
    os.makedirs(exp_dir)
    return exp_dir