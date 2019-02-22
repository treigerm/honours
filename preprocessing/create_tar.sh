#!/bin/bash

set -e

ROOT=/Users/Tim/data/tcga_gbm/sample_slides

METADATA=${ROOT}/metadata
TRAIN=${METADATA}/train_metadata.csv
TEST=${METADATA}/test_metadata.csv
VAL=${METADATA}/val_metadata.csv

OUT_NAME=${ROOT}/top_tiles.tar.gz

cd ${ROOT}
cat <(tail -n +2 ${TRAIN}) <(tail -n +2 ${TEST}) <(tail -n +2 ${VAL}) | \
    cut -d, -f1 | \
    tar -zvcf ${OUT_NAME} -T -