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

def main(embeddings_file, patients_file, out_file):
    with open(embeddings_file, "r") as f:
        embeddings = pickle.load(f) # embeddings: {slide_id: [embeddings]}
    
    with open(patients_file, "r") as f:
        patients2slides = pickle.load(f)

    patients_features = {}
    for patient_id, slide_ids in patients2slides.items():
        patient_embeddings = []
        for slide_id in slide_ids:
            patient_embeddings.append(embeddings[slide_id])

        patient_embeddings = np.array(patient_embeddings)
        patients_features[patient_id] = aggregate_embeddings(patient_embeddings)
    
    with open(out_file, "w+") as f:
        pickle.dump(patients_features, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-file", type=str)
    parser.add_argument("--patients-file", type=str)
    parser.add_argument("--out-file", type=str)
    args = parser.parse_args()
    main(**vars(args))