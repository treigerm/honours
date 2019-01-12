import torch
import argparse
import pickle
import itertools

from models.cae import CAE, TestCAE
from models.factory import get_model

INPUT_SIZE = 1000  # Input slides are 1000x1000
OUTPUT_SIZE = 128  # Output slides should be 128x128

def subslides_indexes():
    ixs = []
    current_ix = 0
    while True:
        if current_ix + OUTPUT_SIZE < INPUT_SIZE:
            ixs.append(current_ix)
            current_ix += OUTPUT_SIZE
        else:
            ixs.append(INPUT_SIZE - OUTPUT_SIZE)
            break
    return ixs

def make_embeddings(model, slides, device):
    embeddings = []
    ixs = itertools.product(subslides_indexes(), subslides_indexes())
    for slide in slides:
        slide = slide.to(device)
        for x_sub_slide, y_sub_slide in ixs:
            embedding = model.encoder(
                slide[x_sub_slide:(x_sub_slide + OUTPUT_SIZE), 
                      y_sub_slide:(y_sub_slide + OUTPUT_SIZE)]
             )
            embeddings.append(embedding.numpy())

    return embeddings

def main(model_name, checkpoint_path, slide_dict_path, out_file, use_gpu):
    device = torch.device("cuda" if use_gpu else "cpu")
    model = get_model(model_name).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    with open(slide_dict_path, "r") as f:
        slide_dict = pickle.load(f)

    embeddings = {}
    for slide_id, slides in slide_dict.items():
        embeddings[slide_id] = make_embeddings(model, slides, device)
    
    with open(out_file, "w+") as f:
        pickle.dump(embeddings, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--slide-dict-path", type=str)
    parser.add_argument("--out-file", type=str)
    parser.add_argument("--use-gpu", type=bool)
    args = parser.parse_args()
    main(**vars(args))