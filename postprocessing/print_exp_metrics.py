#!/usr/bin/env python
import os
import pickle

LOGS_PATH = "/Users/Tim/data/tcga_gbm/logs"
VAL_METRICS_NAME = "val_eval.pickle"
TEST_METRICS_NAME = "test_eval.pickle"

VAL_LOGS = [
    "mil_mean_no_pretrained_20190308-171512",
    "mil_mean_pretrained_20190309-152148",
    "mil_attention_no_pretrained_20190309-111415",
    "mil_attention_pretrained_20190309-122831"
]
TEST_LOGS = [
    "mil_mean_pretrained_20190309-152148",
    "mil_attention_pretrained_20190309-122831"
]

def print_results_table(results):
    print("{:<50} {:<10} {:<10}".format("Name", "Val (in %)", "Test (in %)"))
    for log in results["val"].keys():
        val_acc = results["val"][log]
        test_acc = results["test"][log] if log in results["test"].keys() else -1
        print("{:<50} {:<10.2f} {:<10.2f}".format(log, 100*val_acc, 100*test_acc))

def main():
    results = {
        "val": {},
        "test": {}
    }

    for log in VAL_LOGS:
        metrics_file = os.path.join(LOGS_PATH, log, VAL_METRICS_NAME)
        with open(metrics_file, "rb") as f:
            metric = pickle.load(f)
        results["val"][log] = metric["Accuracy"]
    
    for log in TEST_LOGS:
        metrics_file = os.path.join(LOGS_PATH, log, TEST_METRICS_NAME)
        with open(metrics_file, "rb") as f:
            metric = pickle.load(f)
        results["test"][log] = metric["Accuracy"]

    print_results_table(results)

if __name__ == "__main__":
    main()