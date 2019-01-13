#!/usr/bin/env python

import os
import glob
import csv
import argparse

FIELDNAMES = ["relative_path", "label", "case_id", "section_location"]

def find_tiff_files(root_dir, sub_dir, image_dir=None):
    # TODO: Describe directory structure.
    os.chdir(root_dir)
    # Find all .tiff tiles.
    if image_dir is not None:
        sub_path = os.path.join(sub_dir, image_dir)
    else:
        sub_path = sub_dir

    return glob.glob("{}/*/*.tiff".format(sub_path))

def get_metadata(file_path):
    """
    Filepath of the form: 
        /path/to/TCGA-19-1789-01A-01-BS1.7ce4575c-8bd4-4129-9623-1eb584e7bcff.svs
    
    The first twelve characters indicate the case ID. For the given filepath the 
    case ID would be 'TCGA-19-1789'.

    The 21th character indicates whether the section location is top or bottom.
    For this filepath it would 'B' so bottom.
    """
    metadata = {}
    file_name = os.path.basename(file_path)

    metadata["case_id"] = file_name[:12]

    if file_name[20] == "B":
        metadata["section_location"] = "BOTTOM"
    elif file_name[20] == "T":
        metadata["section_location"] = "TOP"
    else:
        metadata["section_location"] = "UNKNOWN"

    return metadata

def main(root_dir, 
         csv_filename, 
         survival_dir=None, 
         non_survival_dir=None,
         image_dir=None):
    with open(csv_filename, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for label, sub_dir in [(0, non_survival_dir), (1, survival_dir)]:
            for file_path in find_tiff_files(root_dir, sub_dir, image_dir):
                meta = get_metadata(file_path)
                row = {
                    "relative_path": file_path,
                    "label": label,
                    "case_id": meta["case_id"],
                    "section_location": meta["section_location"]
                }
                writer.writerow(row)

if __name__ == "__main__":
    # TODO: Descriptions of arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir")
    parser.add_argument("--csv-filename")
    parser.add_argument("--survival-dir")
    parser.add_argument("--non-survival-dir")
    parser.add_argument("--image-dir")
    args = parser.parse_args()
    main(**vars(args))