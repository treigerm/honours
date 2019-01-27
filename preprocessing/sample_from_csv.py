#!/usr/bin/env python

import pandas as pd
import numpy as np
import argparse
import os
import shutil

RANDOM_STATE = 42

def main(slides_metadata_file, num_samples, root_dir, dest_dir, dry_run):
    slides = pd.read_csv(slides_metadata_file)
    slide_paths = slides["relative_path"].sample(n=num_samples, random_state=RANDOM_STATE)
    for relative_path in slide_paths:
        full_path = os.path.join(root_dir, relative_path)
        if dry_run:
            print("cp {} {}".format(full_path, dest_dir))
        else:
            shutil.copy(full_path, dest_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides-metadata-file", type=str)
    parser.add_argument("--root-dir", type=str)
    parser.add_argument("--dest-dir", type=str)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(**vars(args))