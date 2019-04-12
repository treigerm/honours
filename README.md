# Documentation

The code is organised as follows:

- `data/` stores code to load data
- `models/` stores code to implement the deep learning models
- `notebooks/` has a Jupyter notebook to analyse results of the CAE model
- `postprocessing/` contains scripts to analyse results e.g. plot loss curves
- `preprocessing/` contains scripts to preprocess the data, more information about 
preprocessing can be found in that subdirectory.
- `utils/` contains various utility functions.

The remaining contents of the repository are described below

## Scripts

See `preprocessing/README.md` for how to preprocess the data and in what form 
the data is stored.

Experiments are run with `./run_exp.py --config {CONFIG_FILE}`. `{CONFIG_FILE}`
is a YAML file which specifies the hyperparameters and experiment configurations.
The `sample_config.yaml` is an example config file which shows the possible 
settings. `./sub_run_exp.sh` is the script which can be used to submit a job on 
the Eddie server. The script creates a log directory for each run to save 
checkpoints of the trained model and evaluation results.

The remaining scripts often have common arguments. `--checkpoint-path` is the 
path to a saved model, `--root-dir` is the root directory of the data (see
`preprocessing/README.md`) and `--data-csv` is the CSV file which stores the 
information about the tiles (see `preprocessing/README.md`). The other arguments 
for each script can be found by running `./{SCRIPTNAME}.py --help`.

`./embed_slides.py` is a script to use the CAE encoder part to extract embeddings 
for each tile. It stores the embeddings for each tile in a pickle file. It also 
has the option to directly aggregate the embeddings for each patient.

`./aggregate_embeddings.py` aggregates the tile embeddings for each patient. Works 
with the output pickle file from `./embed_slides.py`.

`./display_samples.py` displays a random sample of the tiles in the dataset. If 
a path to a CAE checkpoint is specified the script also displays the reconstructed 
tiles (i.e. the output of the CAE).

`./extract_important_tiles.py` has an argument `--coef-file` this file specifies 
an importance coefficient to each dimension of the embeddings. The script then 
displays those tiles which have the highest value in the dimension which has the 
highest importance coefficient.

`./mil_att_tiles.py` displays the tiles which receive the highest attention score 
from the MIL model. The script is similar to
`./extract_important_tiles.py` and can possibly be merged into one script. 

`./mil_full_eval.py` evaluates the MIL method on the full dataset.