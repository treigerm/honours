# Preprocessing

The scripts assume that the data is stored on disk in the following structure:

```
root_dir/
    poor_survival_dir/
        images_dir/
            image1_dir/
                wsi1.svs
            image2_dir/
                wsi2.svs
            ...
    good_survival_dir/
        images_dir/
            image1_dir/
                wsi1.svs
            image2_dir/
                wsi2.svs
            ...
```

Our complete preprocessing workflow is as follows:

1. Split each WSI (`.svs` files) into tiles (`.tiff` files) and filter out background tiles
2. Create CSV which stores information about tiles
3. Split CSV into training, validation and testing set

All Python files are command line scripts. For usage information of each script
you can run `./{SCRIPT_NAME} --help`.

## Split WSI and filter tiles

```
./svs2tiles.py --tilesize {TILESIZE} --infile {SVS_FILE} --identifier {IDENTIFIER} --check-neighbours
```

`--identifier` is a string added to the output `.tiff` file to discriminate between 
outputs of different runs. `--check-neighbours` enables the option that a tile is 
only marked as non-background if the upper, lower, left and right neighbour are 
also marked as non-background.

The script can be run in parallel with GNU parallel tool. `./tile_job.sh {DATADIR}` 
is a helper script which can be submitted as a job on the Eddie cluster for 
splitting and filtering an entire dataset.
`./tile_batch.sh` is the script that is used by `./tile_job.sh`.

The output tiles follow the naming convention of
`{prefix}_{x_loc}_{y_loc}_{tilesize}x{tilesize}.{identifier}.tiff`.
`prefix` is the previous filename and `{x_loc}` and `{y_loc}` specify the location 
of the tile in the original WSI.

After running this the images directories will look as follows:

```
images_dir/
    image1_dir/
        wsi1.svs
        tile01_0.identifier.tiff
        tile01.identifier.tiff
        ...
    image2_dir/
        wsi2.svs
        tile01.identifier.tiff
        tile01.identifier.tiff
        ...
    ...
```

## Create CSV

In this step we create a csv which stores the location, label and patient ID of each tile. 

```
./create_csv.py --root-dir {ROOT_DIR} --csv-filename {OUT_FILENAME} --survival-dir {SURVIVAL_DIR} --non-survival-dir {NON_SURVIVAL_DIR} --image-dir {IMAGE_DIR} --identifier {IDENTIFIER}
```

The `--identifier` is from the previous step and the directories correspond to 
the specific directories listed above.

## Split CSV

```
./split_csv.py --train-size {TRAIN_SIZE} --val-size {VAL_SIZE} --slides-metadata-file {CSV_FILE} --out-file {OUT_FILENAME}
```

`{TRAIN_SIZE}` and `{VAL_SIZE}` are floats that specify the fraction of the data 
used for the training set and validation set, respectively. `{CSV_FILE}` gives 
the name CSV file created in the previous step.

The output files of this step can then be used to feed into the PyTorch 
DataLoader class implemented in this repository.

## Other scripts

The other scripts in this directory are one off scripts which I needed at several
different stages of my project.

`create_thumbnail.py` creates thumbnail for `.svs` files.

`sample_from_csv.py` copies a random subsample of the tiles into a target directory.

`normalize.sh` and `stain_normalization.m` are scripts for stain normalisation which 
I did not end up using but are left here for completeness.