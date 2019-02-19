#!/usr/bin/env python

import os
import glob
import argparse
import pprint
from collections import defaultdict
from itertools import groupby

GOOD_LABEL = "good"
POOR_LABEL = "poor"

SUB_DIRS = {
    "good": "TCGA_GBM_60y_to_70y_morethan15mnts_White_n50_biospecimen.cases_selection",
    "poor": "TCGA_GBM_60y_to_70y_lessthan10mnts_White_n45_biospecimen.cases_selection",
}

def get_lowest_true_positive(files, dirs):
    get_id = lambda path: path.split("/")[-3]
    sorted_files = sorted(files, key=get_id)
    groups = groupby(sorted_files, get_id)
    defaults = {d: 0 for d in dirs}
    for k, v in groups:
        defaults[k] = len(list(v))
    return min(defaults.values())

def count_tp_and_fp(root_dir, sub_dir, method_name, image_dir=None):
    os.chdir(root_dir)

    if image_dir is not None:
        sub_path = os.path.join(sub_dir, image_dir)
    else:
        sub_path = sub_dir
    
    all_positives_pattern = "{}/*/{}/*.tiff".format(sub_path, method_name)
    false_positives_pattern = "{}/*/{}/false_positives/*.tiff".format(
        sub_path, method_name
    )

    dirs = [x for x in os.listdir(sub_path) 
            if os.path.isdir(os.path.join(sub_path, x))]

    glob_all_pos = glob.glob(all_positives_pattern)
    all_positives = len(glob_all_pos)
    false_positives = len(glob.glob(false_positives_pattern))
    return {
        "TP": all_positives - false_positives, # True positives
        "FP": false_positives, # False positives
        "LTP": get_lowest_true_positive(glob_all_pos, dirs) # Lowest true positives
    }

def print_stats_table(stats):
    print("{:<10} {:<10} {:<10} {:<10}".format("Method", "TP", "FP", "LTP"))
    for method, vals in stats.items():
        print("-" * 40)
        print("{:<10} {:<10} {:<10} {:<10}".format(method, "", "", ""))
        for label, v in vals.items():
            if label == "total":
                print("")
            print("{0:<10} {TP:<10} {FP:<10} {LTP:<10}".format(label, **v))


def main(root_dir, image_dir, filter_methods):
    stats = defaultdict(dict)
    for label, sub_dir in SUB_DIRS.items():
        for meth in filter_methods:
            stats[meth][label] = count_tp_and_fp(root_dir, sub_dir, meth, image_dir)
    
    for method, vals in stats.items():
        labels = SUB_DIRS.keys()
        stats[method]["total"] = {
            "TP": sum([stats[method][l]["TP"] for l in labels]),
            "FP": sum([stats[method][l]["FP"] for l in labels]),
            "LTP": min([stats[method][l]["LTP"] for l in labels])
            }
    
    print(stats)
    print_stats_table(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir")
    parser.add_argument("--root-dir")
    parser.add_argument("--filter-methods", nargs="+")
    args = parser.parse_args()
    main(**vars(args))