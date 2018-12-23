#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -pe sharedmem 8
#$ -cwd

set -e

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load environment
module load anaconda
source activate honours

SCRIPTSDIR="/home/s1547426/dev/honours/data"

DATADIR="/home/s1547426/data/tcga_gbm"
NUM_CORES=8
TILESIZE=1000

find $DATADIR -mindepth 1 -maxdepth 1 -type d | \
    parallel -j $NUM_CORES ${SCRIPTSDIR}/tile_batch.sh {} $TILESIZE