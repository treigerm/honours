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
    with open(embeddings_file, "rb") as f:
        # dataset: {split_name: {case_id: [(relative_path, embeddings)]}}
        dataset = pickle.load(f) 
    
    for name, embeddings in dataset.items():
        for case_id, case_embeddings in embeddings.items():
            # case_embeddings: [(relative_path, embeddings)]
            case_embeddings = map(lambda x: x[1], case_embeddings)
            embeddings[case_id] = aggregate_embeddings(case_embeddings)
        dataset[name] = embeddings
    
    with open(out_file, "wb+") as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-file", type=str)
    parser.add_argument("--out-file", type=str)
    args = parser.parse_args()
    main(**vars(args))