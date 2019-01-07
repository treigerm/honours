#!/usr/bin/env python

import os
import glob
import csv
import argparse

FIELDNAMES = ["relative_path", "label", "patientID"]

def find_tiff_files(root_dir, sub_dir):
    # TODO: Describe directory structure.
    d = os.path.join(root_dir, sub_dir, "write_access_data_images")
    os.chdir(d)
    # Find all .tiff tiles.
    return glob.glob("*/*.tiff")

def main(root_dir, csv_filename, survival_dir=None, non_survival_dir=None):
    with open(csv_filename, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for label, sub_dir in [(0, non_survival_dir), (1, survival_dir)]:
            for file_path in find_tiff_files(root_dir, sub_dir):
                row = {
                    "relative_path": file_path,
                    "label": label,
                    "patientID": "None"
                }
                writer.writerow(row)

if __name__ == "__main__":
    # TODO: Descriptions of arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir")
    parser.add_argument("--csv-filename")
    parser.add_argument("--survival-dir")
    parser.add_argument("--non-survival-dir")
    args = parser.parse_args()
    main(**vars(args))