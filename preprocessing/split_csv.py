#!/usr/bin/env python

import pandas as pd
import numpy as np
import argparse
import os

RANDOM_SEED = 42

def main(train_size, val_size, slides_metadata_file, out_file):
    slides_metadata = pd.read_csv(slides_metadata_file)
    cases = slides_metadata["case_id"].unique()

    num_cases = len(cases)
    num_train = int(np.floor(train_size * num_cases))
    num_val = int(np.floor(val_size * num_cases))

    rnd = np.random.RandomState(RANDOM_SEED)
    rnd.shuffle(cases)
    train_cases, val_cases, test_cases = np.split(
        cases, [num_train, num_train+num_val]
    )
    splits = [("train", train_cases), ("val", val_cases), ("test", test_cases)]

    out_dir = os.path.dirname(out_file)
    out_basename = os.path.basename(out_file)
    for split_name, cases in splits:
        out = "{}_{}".format(split_name, out_basename)
        out = os.path.join(out_dir, out)
        slides_metadata[slides_metadata["case_id"].isin(cases)].to_csv(out, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=float)
    parser.add_argument("--val-size", type=float)
    parser.add_argument("--slides-metadata-file", type=str)
    parser.add_argument("--out-file", type=str)
    args = parser.parse_args()
    main(**vars(args))