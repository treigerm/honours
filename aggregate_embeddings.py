#!/usr/bin/env python
import pickle
import argparse
import numpy as np

# TODO: Maybe get label information?

def aggregate_embeddings(embeddings):
    """
    Args:
        embeddings (np.array): size of (num_embeddings, num_hidden_dims)
    """
    mean = np.mean(embeddings, axis=0)
    median = np.median(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    return {"mean": mean, "std": std, "median": median}

def main(embeddings_file, out_file):
    with open(embeddings_file, "r") as f:
        embeddings = pickle.load(f) # embeddings: {slide_id: [embeddings]}
    
    for case_id, case_embeddings in embeddings.items():
        embeddings[case_id] = aggregate_embeddings(case_embeddings)
    
    with open(out_file, "w+") as f:
        pickle.dump(embeddings, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-file", type=str)
    parser.add_argument("--out-file", type=str)
    args = parser.parse_args()
    main(**vars(args))