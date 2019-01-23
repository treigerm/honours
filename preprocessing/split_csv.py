#!/usr/bin/env python

import pandas as pd
import numpy as np
import argparse
import os

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def main(train_size, val_size, slides_metadata_file, out_file, section_location):
    slides_metadata = pd.read_csv(slides_metadata_file)
    cases = slides_metadata[["case_id", "label"]].groupby("case_id").agg(lambda x: x.iloc[0])

    train_cases, rest = train_test_split(
        cases, 
        train_size=train_size,
        stratify=cases["label"],
        random_state=RANDOM_SEED
    )
    val_relative_size = val_size / (1.0 - train_size)
    val_cases, test_cases = train_test_split(
        rest,
        train_size=val_relative_size,
        stratify=rest["label"],
        random_state=RANDOM_SEED
    )
    splits = [("train", train_cases), ("val", val_cases), ("test", test_cases)]


    out_dir = os.path.dirname(out_file)
    out_basename = os.path.basename(out_file)
    for split_name, cases in splits:
        out = "{}_{}".format(split_name, out_basename)
        out = os.path.join(out_dir, out)
        mask = slides_metadata["case_id"].isin(cases.index)
        if section_location is not None:
            mask = mask & (slides_metadata["section_location"] == section_location)
        slides_metadata[mask].to_csv(out, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=float)
    parser.add_argument("--val-size", type=float)
    parser.add_argument("--slides-metadata-file", type=str)
    parser.add_argument("--out-file", type=str)
    parser.add_argument("--section-location", type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))